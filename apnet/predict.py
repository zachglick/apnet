"""
Predict interaction energies and atomic properties
"""

import time

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
from apnet import models

@tf.function(experimental_relax_shapes=True)
def predict_monomer_multipoles(model, RA, ZA, mask):
    return model([RA, ZA, mask], training=False)

@tf.function(experimental_relax_shapes=True)
def predict_dimer_sapt(model, feats):
    return model(feats, training=False)

ROOT_DIR = Path(__file__).parent

pair_model_cache = {}
atom_model_cache = {}

default_pair_modelpaths = [f'{ROOT_DIR}/pair_models/hive{i}.h5' for i in range(5)]
default_atom_modelpaths = [f'{ROOT_DIR}/atom_models/hfadz{i+1}.hdf5' for i in range(3)]

nelem = 36
nembed = 10
nnodes = [256,128,64]
nmessage = 3
nrbf= 43
mus = np.linspace(0.8, 5.0, nrbf)
etas = np.array([-100.0] * nrbf)

def load_pair_model(path : str):

    if path not in pair_model_cache:
        if not os.path.isfile(path):
            raise Exception(f'{path} is not a valid path')
        pair_model_cache[path] = models.make_pair_model(nZ=nembed)
        pair_model_cache[path].load_weights(path)
        pair_model_cache[path].call = tf.function(pair_model_cache[path].call, experimental_relax_shapes=True)
    return pair_model_cache[path]

def load_atom_model(path : str, pad_dim : int):

    path_key = (path, pad_dim)
    if path_key not in atom_model_cache:
        print(f"Adding {pad_dim} new")
        if not os.path.isfile(path):
            raise Exception(f'{path} is not a valid path')
        atom_model_cache[path_key] = models.make_atom_model(mus, etas, pad_dim, nelem, nembed, nnodes, nmessage)
        atom_model_cache[path_key].load_weights(path)
        atom_model_cache[path_key].call = tf.function(atom_model_cache[path_key].call, experimental_relax_shapes=True)

    return atom_model_cache[path_key]

def predict_multipoles(molecules, use_ensemble=True):
    """Predict atom-centered multipoles with a pre-trained cartesian-epnn

    Long description

    Parameters
    ----------
    molecules : :class:`~qcelemental.models.Molecule` or list of :class:`~qcelemental.models.Molecule`
        One or more molecules to 

    Returns
    -------
    multipoles : :class:`~numpy.ndarray` or list of :class:`~numpy.ndarray`
        Multipoles q, ux, uy, uz, \Theta_xx, 
    """

    if isinstance(molecules, list):
        molecule_list = [util.qcel_to_monomerdata(molecule) for molecule in molecules]
    else:
        molecule_list = [util.qcel_to_monomerdata(molecules)]

    if use_ensemble:
        atom_modelpaths = default_atom_modelpaths
    else:
        atom_modelpaths = default_atom_modelpaths[:1]

    N = len(molecule_list)

    mtp_prds = []
    mtp_stds = []

    for i, m in enumerate(molecule_list):

        n = len(m[0])
        n_pad = 10 * ((n + 9) // 10)

        # get padded R/Z and total charge mask for CMPNN 
        Rti = np.zeros((1, n_pad, 3))
        Rti[0,:n,:] = m[0] 

        Zti = np.zeros((1, n_pad))
        Zti[0,:n] = m[1] 

        aQti = np.zeros((1, n_pad, n_pad, 1))
        aQti[0,:n,:n,0] = m[2]

        # predict multipoles with CMPNN ensemble
        mtp_prd = []
        
        atom_models = [load_atom_model(path, n_pad) for path in atom_modelpaths]
        for atom_model in atom_models:

            prd = predict_monomer_multipoles(atom_model, Rti, Zti, aQti)
            prd = np.concatenate([np.expand_dims(prd[0], axis=-1), prd[1], prd[2], prd[3]], axis=-1)
            mtp_prd.append(prd)

        mtp_std = np.std(mtp_prd, axis=0)[0,:n,:]
        mtp_prd = np.average(mtp_prd, axis=0)[0,:n,:]

        mtp_stds.append(mtp_std)
        mtp_prds.append(mtp_prd)

    if use_ensemble:
        return mtp_prds, mtp_stds
    else:
        return mtp_prds



def predict_elst(dimers, use_ensemble=True, return_pairs=False):
    """Compute long-range electrostatics from predicted multipoles

    Long description.

    Parameters
    ----------
    dimers : :class:`~qcelemental.models.Molecule` or list of :class:`~qcelemental.models.Molecule`
        One or more dimers to predict the interaction energy of. Each dimer must contain exactly
        two molecular fragments.
    use_ensemble: `bool`, optional
        Do use an ensemble of 3 atom property models? This is more expensive, but improves the 
        prediction accuracy and also provides a prediction uncertainty
    return_pairs: `bool`, optional
        Do return the individual atom-pair electrostatics instead of dimer electrostatics?

    Returns
    -------
    predictions : list of float or list of :class:`numpy.ndarray`
        Predicted long-range electrostatics each dimer in dimers (kcal / mol).
        If return_pairs == False, each prediction is a single float
        If return_pairs == True, each prediction is a numpy.ndarray with shape (NA, NB).
    uncertainties: list of float or list of :class:`numpy.ndarray`
        A prediction uncertainty for each dimer in dimers (kcal / mol). 
        Calculated as the standard deviation of predictions of 3 pretrained atomic property models.
        Has the same length/shape as the returned predictions.
        If use_ensemble == False, this will not be returned.
    """

    if isinstance(dimers, list):
        dimer_list = [util.qcel_to_dimerdata(dimer) for dimer in dimers]
    else:
        dimer_list = [util.qcel_to_dimerdata(dimers)]

    N = len(dimer_list)

    if use_ensemble:
        atom_modelpaths = default_atom_modelpaths
    else:
        atom_modelpaths = default_atom_modelpaths[:1]


    elst_prds = []
    elst_stds = []

    t_start = time.time()

    for i, d in enumerate(dimer_list):

        nA, nB = len(d[0]), len(d[1])
        nA_pad, nB_pad = 10 * ((nA + 9) // 10), 10 * ((nB + 9) // 10)

        # load atom models
        atom_models_A = []
        atom_models_B = []
        for path in atom_modelpaths:
            atom_models_A.append(load_atom_model(path, nA_pad))
            atom_models_B.append(load_atom_model(path, nB_pad))

        # get padded R/Z and total charge mask for CMPNN 
        RAti = np.zeros((1, nA_pad, 3))
        RBti = np.zeros((1, nB_pad, 3))
        RAti[0,:nA,:] = d[0] 
        RBti[0,:nB,:] = d[1] 

        ZAti = np.zeros((1, nA_pad))
        ZBti = np.zeros((1, nB_pad))
        ZAti[0,:nA] = d[2] 
        ZBti[0,:nB] = d[3] 

        aQAti = np.zeros((1, nA_pad, nA_pad, 1))
        aQBti = np.zeros((1, nB_pad, nB_pad, 1))
        aQAti[0,:nA,:nA,0] = d[4]
        aQBti[0,:nB,:nB,0] = d[5]

        # predict multipoles with CMPNN ensemble
        mtp_start = time.time()

        #elst_prds = []
        #pair_prds = []

        elst_prd = []
        
        for atom_model_A, atom_model_B in zip(atom_models_A, atom_models_B):

            mtpA_prd = predict_monomer_multipoles(atom_model_A, RAti, ZAti, aQAti)
            mtpA_prd = np.concatenate([np.expand_dims(mtpA_prd[0], axis=-1), mtpA_prd[1], mtpA_prd[2], mtpA_prd[3]], axis=-1)
            mtpA_prd = mtpA_prd[0,:nA,:]

            mtpB_prd = predict_monomer_multipoles(atom_model_B, RBti, ZBti, aQBti)
            mtpB_prd = np.concatenate([np.expand_dims(mtpB_prd[0], axis=-1), mtpB_prd[1], mtpB_prd[2], mtpB_prd[3]], axis=-1)
            mtpB_prd = mtpB_prd[0,:nB,:]

            tot_prd, pair_prd = multipoles.eval_dimer(d[0], d[1], d[2], d[3], mtpA_prd, mtpB_prd)

            if return_pairs:
                elst_prd.append(pair_prd)
            else:
                elst_prd.append(tot_prd)

        elst_prds.append(np.average(elst_prd, axis=0))
        elst_stds.append(np.std(elst_prd, axis=0))

    if use_ensemble:
        return elst_prds, elst_stds
    else:
        return elst_prds



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
        dimer_list = [util.qcel_to_dimerdata(dimer) for dimer in dimers]
    else:
        dimer_list = [util.qcel_to_dimerdata(dimers)]

    N = len(dimer_list)

    if use_ensemble:
        atom_modelpaths = default_atom_modelpaths
        pair_modelpaths = default_pair_modelpaths
    else:
        atom_modelpaths = default_atom_modelpaths[:1]
        pair_modelpaths = default_pair_modelpaths[:1]

    pair_models = [load_pair_model(path) for path in default_pair_modelpaths]

    sapt_prds = []
    sapt_stds = []

    for i, d in enumerate(dimer_list):

        nA, nB = len(d[0]), len(d[1])
        nA_pad, nB_pad = 10 * ((nA + 9) // 10), 10 * ((nB + 9) // 10)

        # load atom models
        # have to load models inside loop bc model is dependent on natom
        atom_models_A = [load_atom_model(path, nA_pad) for path in atom_modelpaths]
        atom_models_B = [load_atom_model(path, nB_pad) for path in atom_modelpaths]

        # get padded R/Z and total charge mask for CMPNN 
        RAti = np.zeros((1, nA_pad, 3))
        RBti = np.zeros((1, nB_pad, 3))
        RAti[0,:nA,:] = d[0] 
        RBti[0,:nB,:] = d[1] 

        ZAti = np.zeros((1, nA_pad))
        ZBti = np.zeros((1, nB_pad))
        ZAti[0,:nA] = d[2] 
        ZBti[0,:nB] = d[3] 

        aQAti = np.zeros((1, nA_pad, nA_pad, 1))
        aQBti = np.zeros((1, nB_pad, nB_pad, 1))
        aQAti[0,:nA,:nA,0] = d[4]
        aQBti[0,:nB,:nB,0] = d[5]

        # predict multipoles with CMPNN ensemble
        mtpA_prds = []
        mtpB_prds = []
        
        for atom_model in atom_models_A:
            mtpA_prd = predict_monomer_multipoles(atom_model, RAti, ZAti, aQAti)
            mtpA_prd = np.concatenate([np.expand_dims(mtpA_prd[0], axis=-1), mtpA_prd[1], mtpA_prd[2], mtpA_prd[3]], axis=-1)
            mtpA_prds.append(mtpA_prd)

        for atom_model in atom_models_B:
            mtpB_prd = predict_monomer_multipoles(atom_model, RBti, ZBti, aQBti)
            mtpB_prd = np.concatenate([np.expand_dims(mtpB_prd[0], axis=-1), mtpB_prd[1], mtpB_prd[2], mtpB_prd[3]], axis=-1)
            mtpB_prds.append(mtpB_prd)

        mtpA = np.average(mtpA_prds, axis=0)[0,:nA,:]
        mtpB = np.average(mtpB_prds, axis=0)[0,:nB,:]

        # eval elst energy with predicted multipoles
        elst_mtp, pair_mtp = multipoles.eval_dimer(d[0], d[1], d[2], d[3], mtpA, mtpB)

        # make pair features
        features = util.make_features(d[0], d[1], d[2], d[3], mtpA, mtpB, pair_mtp)

        # predict short-range interaction energy
        sapt_prd_pair = np.array([predict_dimer_sapt(pair_model, features) for pair_model in pair_models])

        # add "total interaction energy" dimension
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

    if use_ensemble:
        return sapt_prds, sapt_stds
    else:
        return sapt_prds
