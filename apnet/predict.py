import time
#t_start = time.time()
#print('Loading Libraries...')

import os, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx('float64')

from apnet import multipoles
from apnet import util

#print(f'... Done in {time.time() - t_start:.1f} seconds\n')


@tf.function(experimental_relax_shapes=True)
def predict_monomer_multipoles(model, RA, ZA, mask):
    return model([RA, ZA, mask], training=False)

@tf.function(experimental_relax_shapes=True)
def predict_dimer_sapt(model, feats):
    return model(feats, training=False)

ROOT_DIR = Path(__file__).parent


#print("Loading Models...")
#t_start = time.time()
pad_dim = 26
nelem = 36
nembed = 10
nnodes = [256,128,64]
nmessage = 3
nrbf= 43
mus = np.linspace(0.8, 5.0, nrbf)
etas = np.array([-100.0] * nrbf)
#pair_modelnames = ['badhive', 'badhive', 'badhive']
pair_modelnames = ['badhive']
for pair_modelname in pair_modelnames:
    pair_modelpath = f'{ROOT_DIR}/pair_models/{pair_modelname}.h5'
    if not os.path.isfile(pair_modelpath):
        raise Exception(f'No model exists at {pair_modelpath}')

atom_modelnames = ['hfadz1', 'hfadz2', 'hfadz3']
#atom_modelnames = ['hfadz1']
for atom_modelname in atom_modelnames:
    atom_modelpath = f'{ROOT_DIR}/atom_models/{atom_modelname}.hdf5'
    if not os.path.isfile(atom_modelpath):
        raise Exception(f'No model exists at {atom_modelpath}')

pair_models = []
for pair_modelname in pair_modelnames:
    pair_models.append(util.make_pair_model(nZ=nembed))
    pair_models[-1].load_weights(f'{ROOT_DIR}/pair_models/{pair_modelname}.h5')
    pair_models[-1].call = tf.function(pair_models[-1].call, experimental_relax_shapes=True)

atom_models = []
for atom_modelname in atom_modelnames:
    atom_models.append(util.make_atom_model(mus, etas, pad_dim, nelem, nembed, nnodes, nmessage))
    atom_models[-1].load_weights(f'{ROOT_DIR}/atom_models/{atom_modelname}.hdf5')
    atom_models[-1].call = tf.function(atom_models[-1].call, experimental_relax_shapes=True)
#print(f'... Done in {time.time() - t_start:.1f} seconds\n')




def predict_sapt(qcel_dimers):
    """ Expect list of QCElemental Dimers or single dimer """

    if isinstance(qcel_dimers, list):
        dimers = [util.qcel_to_data(dimer) for dimer in qcel_dimers]
    else:
        dimers = [util.qcel_to_data(qcel_dimers)]

    N = len(dimers)

    mtp_time = 0.0
    elst_time = 0.0
    pair_time = 0.0

    sapt_prds = []

    print('Making Predictions...')
    t_start = time.time()

    for i, d in enumerate(dimers):

        nA, nB = len(d[0]), len(d[1])

        # get padded R/Z and total charge mask for CMPNN 
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

        # predict multipoles with CMPNN ensemble
        mtp_start = time.time()
        mtpA_prds = []
        mtpB_prds = []
        
        for atom_model in atom_models:

            mtpA_prd = predict_monomer_multipoles(atom_model, RAti, ZAti, aQAti)
            mtpA_prd = np.concatenate([np.expand_dims(mtpA_prd[0], axis=-1), mtpA_prd[1], mtpA_prd[2], mtpA_prd[3]], axis=-1)
            mtpA_prds.append(mtpA_prd)

            mtpB_prd = predict_monomer_multipoles(atom_model, RBti, ZBti, aQBti)
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
        pair_start = time.time()
        features = util.make_features(d[0], d[1], d[2], d[3], mtpA, mtpB, pair_mtp)

        # predict short-range interaction energy
        sapt_prd = [predict_dimer_sapt(pair_model, features) for pair_model in pair_models]
        sapt_prd = np.average(sapt_prd, axis=0) # avg ensembles
        sapt_prd = np.sum(sapt_prd, axis=0) # sum pairs
        pair_time += time.time() - pair_start

        sapt_prds.append(sapt_prd)

    print(f'... Done in {time.time() - t_start:.1f} seconds\n')
    print(f'  Multipoles Prediction:    {mtp_time:.1f} seconds')
    print(f'  Multipole Electrostatics: {elst_time:.1f} seconds')
    print(f'  SAPT Prediction:          {pair_time:.1f} seconds')

    sapt_prds = np.concatenate(sapt_prds).reshape(-1,4)
    return sapt_prds


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Use a trained model to predict all four SAPT0 components')

    #parser.add_argument('data',
    #                    help='Dataset for training')
    parser.add_argument('data',
                        help='Dataset for testing')

    args = parser.parse_args(sys.argv[1:])

    data = args.data

    print("Loading Data...")
    t_start = time.time()
    dimers, labels = util.get_dimers(data)
    dimers2 = [util.qcel_to_data(util.data_to_qcel(*dimer)) for dimer in dimers]
    print(f'... Done in {time.time() - t_start:.1f} seconds\n')

    sapt_prds = predict_sapt(dimers)
    sapt_prds = predict_sapt(dimers)

    if labels is not None:
        sapt_errs = sapt_prds - labels
        sapt_maes = np.average(np.absolute(sapt_errs), axis=0)
        print(sapt_maes)
    else:
        for sapt_prd in sapt_prds:
            print(sapt_prd)


    sapt_prds = predict_sapt(dimers2)

    if labels is not None:
        sapt_errs = sapt_prds - labels
        sapt_maes = np.average(np.absolute(sapt_errs), axis=0)
        print(sapt_maes)
    else:
        for sapt_prd in sapt_prds:
            print(sapt_prd)

