"""
train atlas-based alignment with MICCAI2018 version of VoxelMorph, 
specifically adding uncertainty estimation and diffeomorphic transforms.
"""

# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
# tensor callbacks
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.utils import multi_gpu_model 

# project imports
import nibabel as nib
import networks
import losses

sys.path.append('../ext/neuron')
import neuron.callbacks as nrn_gen

def load_volfile(datafile, np_var='vol_data'):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), variable names innp_var (default: 'vol_data')
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        X = nib.load(datafile).get_data()
        
    else: # npz
        if np_var is None:
            np_var = 'vol_data'
        X = np.load(datafile)[np_var]

    return X
def datageneratorsmiccai2018_gen(gen, atlas_vol_bs, batch_size=1, bidir=False):
    """ generator used for miccai 2018 model """
    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)[0]
        if bidir:
            yield ([X, atlas_vol_bs], [atlas_vol_bs, X, zeros])
        else:
            yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros])

def miccai2018_gen_s2s(moving, fixed, zeros,batch_size=1, bidir=False):
    """ generator used for miccai 2018 model """
    while True:
        X = moving
        Y = fixed
        if bidir:
            yield ([X, Y], [Y, X, zeros])
        else:
            yield ([X, Y], [Y, zeros])



def datageneratorsexample_gen(vol_names, batch_size=1, return_segs=False, seg_dir=None, np_var='vol_data'):
    """
    generate examples

    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)

        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
        np_var: specify the name of the variable in numpy files, if your data is stored in 
            npz files. default to 'vol_data'
    """

    while True:
        idxes = np.random.randint(len(vol_names), size=batch_size)

        X_data = []
        for idx in idxes:
            X = load_volfile(vol_names[idx], np_var=np_var)
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)

        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]

        # also return segmentations
        if return_segs:
            X_data = []
            for idx in idxes:
                X_seg = load_volfile(vol_names[idx].replace('norm', 'aseg'), np_var=np_var)
                X_seg = X_seg[np.newaxis, ..., np.newaxis]
                X_data.append(X_seg)
            
            if batch_size > 1:
                return_vals.append(np.concatenate(X_data, 0))
            else:
                return_vals.append(X_data[0])

        yield tuple(return_vals)


def train(atlas_file,
          model_dir,
          gpu_id,
          lr,
          nb_epochs,
          prior_lambda,
          image_sigma,
          steps_per_epoch,
          batch_size,
          load_model_file,
          bidir,
          initial_epoch=0):
    """
    model training function
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model_dir: model folder to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param nb_epochs: number of training iterations
    :param prior_lambda: the prior_lambda, the scalar in front of the smoothing laplacian, in MICCAI paper
    :param image_sigma: the image sigma in MICCAI paper
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param load_model_file: optional h5 model file to initialize with
    :param bidir: logical whether to use bidirectional cost function
    """
    
    # load atlas from provided files. The atlas we used is 160x192x224.
    #oldatlas_vol = np.load('../data/atlas_norm.npz')['vol'][np.newaxis, ..., np.newaxis]
    #oldmov = np.load('../data/test_vol.npz')['vol_data'][np.newaxis, ..., np.newaxis]
    fix_nii = nib.load(atlas_file)
    atlas_vol = fix_nii.get_data()[np.newaxis, ..., np.newaxis]
    vol_size = atlas_vol.shape[1:-1] 
    myzeros  = np.zeros((batch_size, *vol_size, len(vol_size)))
    # prepare data files
    # for the CVPR and MICCAI papers, we have data arranged in train/validate/test folders
    # inside each folder is a /vols/ and a /asegs/ folder with the volumes
    # and segmentations. All of our papers use npz formated data.
    #train_vol_names = glob.glob(os.path.join(data_dir, '*.npz'))
    movingfile = 'mydata/dynamic.0000.nii.gz'
    train_vol_names = [movingfile]
    random.shuffle(train_vol_names)  # shuffle volume list
    assert len(train_vol_names) == 1, "Could not find any training data"

    mov_nii = nib.load(movingfile )
    mov = mov_nii.get_data()[np.newaxis, ..., np.newaxis]

    # Diffeomorphic network architecture used in MICCAI 2018 paper
    nf_enc = [16,32,32,32]
    nf_dec = [32,32,32,32,16,3]

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # gpu handling
    gpu = '/gpu:%d' % 0 # gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # prepare the model
    with tf.device(gpu):
        # the MICCAI201 model takes in [image_1, image_2] and outputs [warped_image_1, velocity_stats]
        # in these experiments, we use image_2 as atlas
        model = networks.miccai2018_net(vol_size, nf_enc, nf_dec, bidir=bidir)

        # load initial weights
        if load_model_file is not None and load_model_file != "":
            model.load_weights(load_model_file)

        # save first iteration
        model.save(os.path.join(model_dir, '%02d.h5' % initial_epoch))

        # compile
        # note: best to supply vol_shape here than to let tf figure it out.
        flow_vol_shape = model.outputs[-1].shape[1:-1]
        loss_class = losses.Miccai2018(image_sigma, prior_lambda, flow_vol_shape=flow_vol_shape)
        if bidir:
            model_losses = [loss_class.recon_loss, loss_class.recon_loss, loss_class.kl_loss]
            loss_weights = [0.5, 0.5, 1]
        else:
            #model_losses = [loss_class.recon_loss, loss_class.kl_loss]
            model_losses = [ losses.NCC().loss   , losses.Grad('l2').loss]
            loss_weights = [1, 1]
        #tf_mov   = tf.constant(mov)
        #tf_fix   = tf.constant(atlas_vol)
        #tf_zeros = tf.constant(myzeros)
        tf_mov   = mov
        tf_fix   = atlas_vol
        tf_zeros = myzeros
        
    
    # data generator
    nb_gpus = len(gpu_id.split(','))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)

    #train_example_gen = datageneratorsexample_gen(train_vol_names, batch_size=batch_size)
    #atlas_vol_bs = np.repeat(atlas_vol, batch_size, axis=0)
    miccai2018_gen = miccai2018_gen_s2s(tf_mov , tf_fix ,tf_zeros ,
                                                   batch_size=batch_size,
                                                   bidir=bidir)

    # prepare callbacks
    save_file_name = os.path.join(model_dir, '{epoch:02d}.h5')

    # fit generator
    with tf.device(gpu):

        # multi-gpu support
        if nb_gpus > 1:
            save_callback = nrn_gen.ModelCheckpointParallel(save_file_name)
            mg_model = multi_gpu_model(model, gpus=nb_gpus)
        
        # single gpu
        else:
            save_callback = ModelCheckpoint(save_file_name)
            # tensorboard --logdir='s2slog' --port=6010
            tensorboard = TensorBoard(log_dir='s2slog', histogram_freq=0, write_graph=True, write_images=False)
            mg_model = model

        mg_model.compile(optimizer=Adam(lr=lr), loss=model_losses, loss_weights=loss_weights)
        mg_model.fit_generator(miccai2018_gen, 
                               initial_epoch=initial_epoch,
                               epochs=nb_epochs,
                               callbacks=[tensorboard,save_callback],
                               steps_per_epoch=steps_per_epoch,
                               verbose=1)
        # register
        [moved, warp] = mg_model.predict([mov, atlas_vol ])

    # output image
    out_img= 'myout.nii.gz'
    if out_img is not None:
        img = nib.Nifti1Image(moved[0,...,0], mov_nii.affine)
        nib.save(img, out_img)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--atlas_file", type=str,
                        dest="atlas_file", default='mydata/dynamic.0033.nii.gz',
                        help="gpu id number")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='s2smodels/',
                        help="models folder")
    parser.add_argument("--gpu", type=str, default='0',
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-6, help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=500,
                        help="number of iterations")
    parser.add_argument("--prior_lambda", type=float,
                        dest="prior_lambda", default=10,
                        help="prior_lambda regularization parameter")
    parser.add_argument("--image_sigma", type=float,
                        dest="image_sigma", default=0.02,
                        help="image noise parameter")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=100,
                        help="frequency of model saves")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=1,
                        help="batch_size")
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default=None,
                        help="optional h5 model file to initialize with")
    parser.add_argument("--bidir", type=int,
                        dest="bidir", default=0,
                        help="whether to use bidirectional cost function")
    parser.add_argument("--initial_epoch", type=int,
                        dest="initial_epoch", default=0,
                        help="first epoch")

    args = parser.parse_args()
    print(args)
    train(**vars(args))
