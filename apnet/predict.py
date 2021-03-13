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
pair_modelnames = ['hive0', 'hive1', 'hive2', 'hive3', 'hive4']
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

pair_models_all = []
for pair_modelname in pair_modelnames:
    pair_models_all.append(util.make_pair_model(nZ=nembed))
    pair_models_all[-1].load_weights(f'{ROOT_DIR}/pair_models/{pair_modelname}.h5')
    pair_models_all[-1].call = tf.function(pair_models_all[-1].call, experimental_relax_shapes=True)

atom_models_all = []
for atom_modelname in atom_modelnames:
    atom_models_all.append(util.make_atom_model(mus, etas, pad_dim, nelem, nembed, nnodes, nmessage))
    atom_models_all[-1].load_weights(f'{ROOT_DIR}/atom_models/{atom_modelname}.hdf5')
    atom_models_all[-1].call = tf.function(atom_models_all[-1].call, experimental_relax_shapes=True)
#print(f'... Done in {time.time() - t_start:.1f} seconds\n')




def predict_sapt(dimers, use_ensemble=True, return_pairs=False):
    """Predict interaction energies with a pre-trained AP-Net model

    Predicts the interaction energy (at the SAPT0/aug-cc-pV(D+d)Z level of theory) for one or
    more molecular dimers. Predictions are decomposed into physically meaningful SAPT components.
    (electrostatics, exchange, induction, and dispersion).


    Parameters
    ----------
    dimers : :class:`~qcelemental.models.Molecule` or list of :class:`~qcelemental.models.Molecule`
        One or more dimers to predict the interaction energy of. Each dimer must contain exactly
        two molecular fragments.
    use_ensemble: `bool`, optional
        Do use an ensemble of 5 AP-Net models? This is more expensive, but improves the prediction
        accuracy and also provides a prediction uncertainty
    return_pairs: `bool`, optional
        Do return the individual atom-pair interaction energies instead of dimer interaction
        energies?

    Returns
    -------
    predictions : :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
        A predicted SAPT interaction energy breakdown for each dimer in dimers (kcal / mol).
        If return_pairs == False, each prediction is a numpy.ndarray with shape (5,).
        If return_pairs == True, each prediction is a numpy.ndarray with shape (5, NA, NB).
        The first dimension of length 5 indexes SAPT components:
        [total, electrostatics, exchange, induction, dispersion]
    uncertainties: :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
        A prediction uncertainty for each dimer in dimers (kcal / mol). 
        Calculated as the standard deviation of predictions of 5 pretrained AP-Net models.
        Has the same length/shape as the returned predictions.
        If use_ensemble == False, this will not be returned.
    """

    if isinstance(dimers, list):
        dimer_list = [util.qcel_to_data(dimer) for dimer in dimers]
    else:
        dimer_list = [util.qcel_to_data(dimers)]

    if use_ensemble:
        pair_models = pair_models_all
        atom_models = atom_models_all
    else:
        pair_models = pair_models_all[:1]
        atom_models = atom_models_all[:1]

    N = len(dimer_list)

    mtp_time = 0.0
    elst_time = 0.0
    pair_time = 0.0

    sapt_prds = []
    sapt_stds = []

    #print('Making Predictions...')
    t_start = time.time()

    for i, d in enumerate(dimer_list):

        nA, nB = len(d[0]), len(d[1])
        if nA > pad_dim or nB > pad_dim:
            raise AssertionError(f"Monomer too large (for now), must be no more than {pad_dim} atoms")

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
        # nensemble x npair x 4
        sapt_prd_pair = np.array([predict_dimer_sapt(pair_model, features) for pair_model in pair_models])
        sapt_prd_pair = np.concatenate([np.sum(sapt_prd_pair, axis=2, keepdims=True), sapt_prd_pair], axis=2)

        if return_pairs:
            sapt_prd = np.transpose(sapt_prd_pair, axes=(0,2,1))
            sapt_prd = sapt_prd.reshape(-1,5,nA,nB) # nensemble x 5 x nA x nB
        else:
            sapt_prd = np.sum(sapt_prd_pair, axis=1) # nensemble x 5

        # ensemble std and avg
        sapt_std = np.std(sapt_prd, axis=0)
        sapt_prd = np.average(sapt_prd, axis=0)

        sapt_prds.append(sapt_prd)
        sapt_stds.append(sapt_std)

        pair_time += time.time() - pair_start

    #print(f'... Done in {time.time() - t_start:.1f} seconds\n')
    #print(f'  Multipoles Prediction:    {mtp_time:.1f} seconds')
    #print(f'  Multipole Electrostatics: {elst_time:.1f} seconds')
    #print(f'  SAPT Prediction:          {pair_time:.1f} seconds')

    if use_ensemble:
        return sapt_prds, sapt_stds
    else:
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

