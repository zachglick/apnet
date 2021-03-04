import time
t_start = time.time()
print('Loading Libraries...')

import os, sys, argparse
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
#tf.keras.backend.set_learning_phase(0)

import multipoles
import util

print(f'... Done in {time.time() - t_start:.1f} seconds\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Use a trained model to predict all four SAPT0 components')

    #parser.add_argument('data',
    #                    help='Dataset for training')
    parser.add_argument('data',
                        help='Dataset for testing')

    #  model name
    parser.add_argument('pair_modelname',
                        help='Existing model to use for inference.')

    # optional arguments: feature hyperparameters
    # (used to ensure correct model reconstruction, these ought to be serialized and saved w the model)
    parser.add_argument('--acsf_nmu',
                        help='ACSF hyperparameter (number of radial gaussians).',
                        type=int,
                        default=43)
    parser.add_argument('--apsf_nmu',
                        help='APSF hyperparameter (number of angular gaussians).',
                        type=int,
                        default=21)
    parser.add_argument('--acsf_eta',
                        help='ACSF hyperparameter (radial gaussian width).',
                        type=int,
                        default=100)
    parser.add_argument('--apsf_eta',
                        help='APSF hyperparameter (angular gaussian width).',
                        type=int,
                        default=25)

    parser.add_argument('--batch_size',
                        help='Batch size for inference (number of dimers)',
                        type=int,
                        default=8)

    args = parser.parse_args(sys.argv[1:])

    data = args.data
    pair_modelname = args.pair_modelname
    batch_size = args.batch_size
    ACSF_nmu = args.acsf_nmu
    APSF_nmu = args.apsf_nmu
    ACSF_eta = args.acsf_eta
    APSF_eta = args.apsf_eta

    pad_dim = 26
    nelem = 36
    
    # this is up to the user
    nembed = 10
    nnodes = [256,128,64]
    nmessage = 3

    # make the model 
    nrbf= 43
    mus = np.linspace(0.8, 5.0, nrbf)
    etas = np.array([-100.0] * nrbf)

    # ensure valid model path
    pair_modelpath = f'./pair_models/{pair_modelname}.h5'
    if not os.path.isfile(pair_modelpath):
        raise Exception(f'No model exists at {pair_modelpath}')

    atom_modelnames = ['hfadz1', 'hfadz2', 'hfadz3']
    for atom_modelname in atom_modelnames:
        if not os.path.isfile(f'atom_models/{atom_modelname}.hdf5'):
            raise Exception(f'No model exists at atom_models/{atom_modelname}.hdf5')

    print("Loading Models...")
    t_start = time.time()
    pair_model = util.make_model(nZ=nembed)
    pair_model.load_weights(f'{pair_modelpath}')
    
    atom_models = []
    for atom_modelname in atom_modelnames:
        atom_models.append(util.make_atom_model(mus, etas, pad_dim, nelem, nembed, nnodes, nmessage))
        atom_models[-1].load_weights(f'./atom_models/{atom_modelname}.hdf5')
        #atom_models[-1].call = tf.function(atom_models[-1].call, experimental_relax_shapes=True)
    print(f'... Done in {time.time() - t_start:.1f} seconds\n')

    print("Loading Data...")
    t_start = time.time()
    dimer, label = util.get_dimers(data)
    N = len(dimer)
    print(f'... Done in {time.time() - t_start:.1f} seconds\n')

    pad_time = 0.0
    mtp_time = 0.0
    elst_time = 0.0
    feat_time = 0.0
    pair_time = 0.0

    yt_prds = []
    yt_errs = []

    print('Making Predictions...')
    t_start = time.time()

    for i, d in enumerate(dimer):

        nA, nB = len(d[0]), len(d[1])

        # get padded R/Z and total charge mask for CMPNN 
        pad_start = time.time()
        RAti = np.zeros((1, pad_dim, 3))
        RBti = np.zeros((1, pad_dim, 3))
        RAti[0,:nA,:] = d[0] 
        RBti[0,:nB,:] = d[1] 

        ZAti = np.zeros((1, pad_dim))
        ZBti = np.zeros((1, pad_dim))
        ZAti[0,:nA] = d[2] 
        ZBti[0,:nB] = d[3] 

        aQAti = np.zeros((1, pad_dim, pad_dim, 1))
        aQBti = np.zeros((1, pad_dim, pad_dim, 1))
        aQAti[0,:nA,:nA,0] = d[4]
        aQBti[0,:nB,:nB,0] = d[5]
        pad_time += time.time() - pad_start

        # predict multipoles with CMPNN ensemble
        mtp_start = time.time()
        mtpA_prds = []
        mtpB_prds = []
        
        for atom_model in atom_models:

            mtpA_prd = util.predict_single(atom_model, [RAti, ZAti, aQAti])
            mtpA_prd = np.concatenate([np.expand_dims(mtpA_prd[0], axis=-1), mtpA_prd[1], mtpA_prd[2], mtpA_prd[3]], axis=-1)
            mtpA_prds.append(mtpA_prd)

            mtpB_prd = util.predict_single(atom_model, [RBti, ZBti, aQBti])
            mtpB_prd = np.concatenate([np.expand_dims(mtpB_prd[0], axis=-1), mtpB_prd[1], mtpB_prd[2], mtpB_prd[3]], axis=-1)
            mtpB_prds.append(mtpB_prd)

        mtpA = np.average(mtpA_prds, axis=0)
        mtpB = np.average(mtpB_prds, axis=0)

        mtpA = mtpA[0,:nA,:]
        mtpB = mtpB[0,:nB,:]
        mtp_time += time.time() - mtp_start

        # eval elst energy with predicted multipoles
        elst_start = time.time()
        elst_mtp, pair_mtp = multipoles.eval_dimer(d[0], d[1], d[2], d[3], mtpA, mtpB)
        elst_time += time.time() - elst_start

        # make pair features
        feat_start = time.time()
        features = util.make_features(d[0], d[1], d[2], d[3], mtpA, mtpB, pair_mtp)
        feat_time += time.time() - feat_start

        # predict short-range interaction energy
        pair_start = time.time()
        sapt_prd = util.predict_single(pair_model, features)
        sapt_prd = np.sum(sapt_prd, axis=0)
        pair_time += time.time() - pair_start

        yt_prds.append(sapt_prd)
        yt_errs.append(sapt_prd - label[i])

    print(f'... Done in {time.time() - t_start:.1f} seconds\n')
    print(f'  Padding:        {pad_time:.1f} seconds')
    print(f'  Multipoles:     {mtp_time:.1f} seconds')
    print(f'  Electrostatics: {elst_time:.1f} seconds')
    print(f'  Features:       {feat_time:.1f} seconds')
    print(f'  SAPT:           {pair_time:.1f} seconds')

    yt_errs = np.concatenate(yt_errs).reshape(-1,4)
    yt_prds = np.concatenate(yt_prds).reshape(-1,4)
    yt_maes = np.average(np.absolute(yt_errs), axis=0)

    print(yt_maes)
