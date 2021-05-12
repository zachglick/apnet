"""
Predict interaction energies and atomic properties
"""

import time
import os 
from pathlib import Path
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx('float64')

import qcelemental as qcel

from apnet import multipoles
from apnet import util
from apnet import models

@tf.function(experimental_relax_shapes=True)
def predict_monomer_multipoles(model, RA, ZA, mask):
    return model([RA, ZA, mask], training=False)

@tf.function(experimental_relax_shapes=True)
def predict_monomer_properties(model, RA, ZA, mask):
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
nrbf = 43
napsf = 21
mus = np.linspace(0.8, 5.0, nrbf)
etas = np.array([-100.0] * nrbf)

class Timer:

   def __init__(self):

       self.durations = {}
       self.starts = {}

   def start(self, name):

       if name in self.starts:
           raise Exception(f"Timer error: '{name}' already started")

       self.starts[name] = time.time()

   def stop(self, name):

       if name not in self.starts:
           raise Exception(f"Timer error: '{name}' does not exist")

       dt = time.time() - self.starts[name]
       self.starts.pop(name)

       if name in self.durations:
           self.durations[name] += dt
       else:
           self.durations[name] = dt

   def print(self):

       print("\nTimings:")
       for k, v in self.durations.items():
           print(f"  {k:30s} {v:8.2f}")
       print()


def load_pair_model(path : str):

    if path not in pair_model_cache:
        if not os.path.isfile(path):
            raise Exception(f'{path} is not a valid path')
        pair_model_cache[path] = models.make_pair_model(nZ=nembed, ACSF_nmu=nrbf, APSF_nmu=napsf)
        pair_model_cache[path].load_weights(path)
        pair_model_cache[path].call = tf.function(pair_model_cache[path].call, experimental_relax_shapes=True)
    return pair_model_cache[path]

def load_atom_model(path : str, pad_dim : int):

    path_key = (path, pad_dim)
    if path_key not in atom_model_cache:
        if not os.path.isfile(path):
            raise Exception(f'{path} is not a valid path')
        try:
            atom_model_cache[path_key] = models.make_atom_model(mus, etas, pad_dim, nelem, nembed, nnodes, nmessage, do_properties=True)
            atom_model_cache[path_key].load_weights(path)
            atom_model_cache[path_key].call = tf.function(atom_model_cache[path_key].call, experimental_relax_shapes=True)
        except:
            atom_model_cache[path_key] = models.make_atom_model(mus, etas, pad_dim, nelem, nembed, nnodes, nmessage, do_properties=False)
            atom_model_cache[path_key].load_weights(path)
            atom_model_cache[path_key].call = tf.function(atom_model_cache[path_key].call, experimental_relax_shapes=True)

    return atom_model_cache[path_key]

def predict_multipoles(molecules, modelpath=None, use_ensemble=True):
    """Predict atom-centered multipoles with a pre-trained model

    Predicts atomic charge distributions (at the HF/aug-cc-pV(D+d)Z level of theory) in the form
    of atom-centered multipoles (charge, dipole, and quadrupole). Predicted atomic charges are 
    guaranteed to sum to the total charge, and quadrupoles are guaranteed to be traceless.

    Parameters
    ----------
    molecules : :class:`~qcelemental.models.Molecule` or list of :class:`~qcelemental.models.Molecule`
        One or more molecules to predict the atomic multipoles of. Each molecule must contain exactly
        one molecular fragment.
    modelpath : `str`, optional
        Path to file of atomic property model weights to be used in prediction. If not specified,
        the pre-trained HF/aDZ model will be used.
    use_ensemble : `bool`, optional
        Do use an ensemble of pre-trained atomic property models? If modelpath is specified, this
        parameter is ignored.

    Returns
    -------
    predictions : list of :class:`~numpy.ndarray`
        The multipole predictions for each molecule are an array of shape (NATOM, 10).
        The following ordering is used for the second index: [charge, dipole_x, dipole_y, dipole_z,
        quadrupole_xx, quadrupole_xy, quadrupole_xz, quadrupole_yy, quadrupole_yz, quadrupole_zz].
    uncertainties : list of :class:`~numpy.ndarray`
        A prediction uncertainty for each molecule in molecules (kcal / mol). 
        Calculated as the standard deviation of predictions of 3 pretrained atomic property models.
        Has the same length/shape as the returned predictions.
        Only returned if use_ensemble == True and modelpath == None.
    """

    if isinstance(molecules, list):
        molecule_list = [util.qcel_to_monomerdata(molecule) for molecule in molecules]
    else:
        molecule_list = [util.qcel_to_monomerdata(molecules)]

    if modelpath is not None:
        atom_modelpaths = [modelpath]
    else:
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

    if len(atom_modelpaths) > 1:
        return mtp_prds, mtp_stds
    else:
        return mtp_prds



def predict_elst(dimers, use_ensemble=True, return_pairs=False):
    """Predict long-range electrostatics interactions with a pre-trained model

    Predicts the long-range electrostatic component of the interaction energy (at the SAPT0/aug-cc-pV(D+d)Z 
    level of theory) for one or more molecular dimers. This is done by predicting atom-centered charges,
    dipoles, and quadrupoles on each monomer and evaluating the electrostatic interactions energy of the 
    predicted multipoles.

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
        A predicted long-range electrostatics for each dimer (kcal / mol).
        If return_pairs == False, each prediction is a single float.
        If return_pairs == True, each prediction is a numpy.ndarray with shape (NA, NB).
    uncertainties: list of float or list of :class:`numpy.ndarray`
        A prediction uncertainty for each dimer (kcal / mol). 
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
    """Predict interaction energies with a pre-trained model

    Predicts the interaction energy (at the SAPT0/aug-cc-pV(D+d)Z level of theory) for one or
    more molecular dimers. Predictions are decomposed into physically meaningful SAPT components.
    (electrostatics, exchange, induction, and dispersion). Electrostatic predictions include
    both short-range charge penetration effects and long-range multipole electrostatics.

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
        A predicted SAPT interaction energy breakdown for each dimer (kcal / mol).
        If return_pairs == False, each prediction is a numpy.ndarray with shape (5,).
        If return_pairs == True, each prediction is a numpy.ndarray with shape (5, NA, NB).
        The first dimension of length 5 indexes SAPT components:
        [total, electrostatics, exchange, induction, dispersion].
    uncertainties: :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
        A prediction uncertainty for each dimer (kcal / mol). 
        Calculated as the standard deviation of predictions of 5 pretrained models.
        Has the same length/shape as the returned predictions.
        If use_ensemble == False, this will not be returned.
    """

    pred_timer = Timer()

    pred_timer.start('Total Prediction')
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

    pred_timer.start('Loading Pair Models')
    pair_models = [load_pair_model(path) for path in default_pair_modelpaths]
    pred_timer.stop('Loading Pair Models')

    sapt_prds = []
    sapt_stds = []

    for i, d in enumerate(dimer_list):

        nA, nB = len(d[0]), len(d[1])
        nA_pad, nB_pad = 10 * ((nA + 9) // 10), 10 * ((nB + 9) // 10)

        # load atom models
        # have to load models inside loop bc model is dependent on natom
        pred_timer.start('Loading Atom Models')
        atom_models_A = [load_atom_model(path, nA_pad) for path in atom_modelpaths]
        atom_models_B = [load_atom_model(path, nB_pad) for path in atom_modelpaths]
        pred_timer.stop('Loading Atom Models')

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
        
        pred_timer.start('Predicting Multipoles')
        for atom_model in atom_models_A:
            mtpA_prd = predict_monomer_multipoles(atom_model, RAti, ZAti, aQAti)
            mtpA_prd = np.concatenate([np.expand_dims(mtpA_prd[0], axis=-1), mtpA_prd[1], mtpA_prd[2], mtpA_prd[3]], axis=-1)
            mtpA_prds.append(mtpA_prd)

        for atom_model in atom_models_B:
            mtpB_prd = predict_monomer_multipoles(atom_model, RBti, ZBti, aQBti)
            mtpB_prd = np.concatenate([np.expand_dims(mtpB_prd[0], axis=-1), mtpB_prd[1], mtpB_prd[2], mtpB_prd[3]], axis=-1)
            mtpB_prds.append(mtpB_prd)
        pred_timer.stop('Predicting Multipoles')

        mtpA = np.average(mtpA_prds, axis=0)[0,:nA,:]
        mtpB = np.average(mtpB_prds, axis=0)[0,:nB,:]

        # eval elst energy with predicted multipoles
        pred_timer.start('Electrostatics Evaluation')
        elst_mtp, pair_mtp = multipoles.eval_dimer(d[0], d[1], d[2], d[3], mtpA, mtpB)
        pred_timer.stop('Electrostatics Evaluation')

        # make pair features
        pred_timer.start('Pair Features')
        features = util.make_features(d[0], d[1], d[2], d[3], mtpA, mtpB, pair_mtp)
        pred_timer.stop('Pair Features')

        # predict short-range interaction energy
        pred_timer.start('Predicting Pair Energies')
        sapt_prd_pair = np.array([predict_dimer_sapt(pair_model, features) for pair_model in pair_models])
        pred_timer.stop('Predicting Pair Energies')

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

    pred_timer.stop('Total Prediction')
    pred_timer.print()

    if use_ensemble:
        return sapt_prds, sapt_stds
    else:
        return sapt_prds



def predict_sapt_common(common_monomer, monomers, use_ensemble=True, return_pairs=False):
    """Predict interaction energies with a pre-trained model

    Predicts the interaction energy (at the SAPT0/aug-cc-pV(D+d)Z level of theory) for one or
    more molecular dimers. Predictions are decomposed into physically meaningful SAPT components.
    (electrostatics, exchange, induction, and dispersion). Electrostatic predictions include
    both short-range charge penetration effects and long-range multipole electrostatics.
    
    This is a special case of `apnet.predict_sapt` where one monomer is common to all dimers.
    This method is more efficient, particularly when the common monomer is large. An example
    use case is assessing the interaction energy between a (large) protein pocket, and many docked
    ligands.

    Parameters
    ----------
    common_monomer: :class:`~qcelemental.models.Molecule`
        A single monomer (one molecular fragment) that is the same across all dimers. (For example,
        a protein pocket)
    monomers: list of :class:`~qcelemental.models.Molecule`
        One or more monomers (one molecular fragment) for which the interaction energy between each
        monomer and `common_monomer` will be predicted. (For example, a list of docked ligands)
    use_ensemble: `bool`, optional
        Do use an ensemble of 5 AP-Net models? This is more expensive, but improves the prediction
        accuracy and also provides a prediction uncertainty
    return_pairs: `bool`, optional
        Do return the individual atom-pair interaction energies instead of dimer interaction
        energies?

    Returns
    -------
    predictions : :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
        A predicted SAPT interaction energy breakdown for each dimer (kcal / mol).
        If return_pairs == False, each prediction is a numpy.ndarray with shape (5,).
        If return_pairs == True, each prediction is a numpy.ndarray with shape (5, NA, NB).
        The first dimension of length 5 indexes SAPT components:
        [total, electrostatics, exchange, induction, dispersion].
    uncertainties: :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
        A prediction uncertainty for each dimer (kcal / mol). 
        Calculated as the standard deviation of predictions of 5 pretrained models.
        Has the same length/shape as the returned predictions.
        If use_ensemble == False, this will not be returned.
    """

    if not isinstance(common_monomer, qcel.models.Molecule):
        raise Exception(f'Argument `common_monomer` is not a Molecule')

    if not isinstance(monomers, list):
        raise Exception(f'Argument `monomers` should be a List of Molecule objects')

    for monomer in monomers:
        if not isinstance(monomer, qcel.models.Molecule):
            raise Exception(f'Argument `monomers` should be a List of Molecule objects')

    monomerA = util.qcel_to_monomerdata(common_monomer)
    monomerB_list = [util.qcel_to_monomerdata(monomer) for monomer in monomers]

    N = len(monomers)

    if use_ensemble:
        atom_modelpaths = default_atom_modelpaths
        pair_modelpaths = default_pair_modelpaths
    else:
        atom_modelpaths = default_atom_modelpaths[:1]
        pair_modelpaths = default_pair_modelpaths[:1]

    pair_models = [load_pair_model(path) for path in default_pair_modelpaths]

    nA = len(monomerA[0])
    atom_models_A = [load_atom_model(path, nA) for path in atom_modelpaths]

    # get padded R/Z and total charge mask for CMPNN 
    RAti = np.zeros((1, nA, 3))
    RAti[0,:nA,:] = monomerA[0] 
    ZAti = np.zeros((1, nA))
    ZAti[0,:nA] = monomerA[1] 
    aQAti = np.zeros((1, nA, nA, 1))
    aQAti[0,:nA,:nA,0] = monomerA[2]

    # predict multipoles with CMPNN ensemble
    mtpA_prds = []

    for atom_model in atom_models_A:
        mtpA_prd = predict_monomer_multipoles(atom_model, RAti, ZAti, aQAti)
        mtpA_prd = np.concatenate([np.expand_dims(mtpA_prd[0], axis=-1), mtpA_prd[1], mtpA_prd[2], mtpA_prd[3]], axis=-1)
        mtpA_prds.append(mtpA_prd)

    mtpA = np.average(mtpA_prds, axis=0)[0,:nA,:]

    sapt_prds = []
    sapt_stds = []

    for i, monomerB in enumerate(monomerB_list):

        nB = len(monomerB[0])
        nB_pad =  10 * ((nB + 9) // 10)

        # load atom models
        # have to load models inside loop bc model is dependent on natom
        atom_models_B = [load_atom_model(path, nB_pad) for path in atom_modelpaths]

        # get padded R/Z and total charge mask for CMPNN 
        RBti = np.zeros((1, nB_pad, 3))
        RBti[0,:nB,:] = monomerB[0] 
        ZBti = np.zeros((1, nB_pad))
        ZBti[0,:nB] = monomerB[1] 
        aQBti = np.zeros((1, nB_pad, nB_pad, 1))
        aQBti[0,:nB,:nB,0] = monomerB[2]

        # predict multipoles with CMPNN ensemble
        mtpB_prds = []
        
        for atom_model in atom_models_B:
            mtpB_prd = predict_monomer_multipoles(atom_model, RBti, ZBti, aQBti)
            mtpB_prd = np.concatenate([np.expand_dims(mtpB_prd[0], axis=-1), mtpB_prd[1], mtpB_prd[2], mtpB_prd[3]], axis=-1)
            mtpB_prds.append(mtpB_prd)

        mtpB = np.average(mtpB_prds, axis=0)[0,:nB,:]

        # eval elst energy with predicted multipoles
        elst_mtp, pair_mtp = multipoles.eval_dimer(monomerA[0], monomerB[0], monomerA[1], monomerB[1], mtpA, mtpB)

        # make pair features
        features = util.make_features(monomerA[0], monomerB[0], monomerA[1], monomerB[1], mtpA, mtpB, pair_mtp)

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

def predict_cliff_properties(molecules, modelpath):
    """Predict atomic properties for the Component Learned Intermolecular Force Field (CLIFF)

    Using a trained neural network stored at `modelpath`, predicts atomic charge distributions in
    the form of atom-centered multipoles (charge, dipole, and quadrupole), atomic volume ratios,
    and the exponential width of atomic valence electron density. Predicted atomic charges are
    guaranteed to sum to the total charge, and quadrupoles are guaranteed to be traceless.

    Parameters
    ----------
    molecules : list of :class:`~qcelemental.models.Molecule`
        One or more molecules to predict the atomic multipoles of. Each molecule must contain exactly
        one molecular fragment.
    modelpath : `str`, optional
        Path to file of atomic property model weights to be used in prediction.

    Returns
    -------
    predictions : list of :class:`dict`
        A separate `dict` is returned for each molecule in `molecules`. Each `dict` contains the
        following keys, all of which map to a :class:`numpy.ndarray` with the following shape:

          - charge : (NATOM)
          - dipoles : (NATOM, 3)
          - quadrupoles : (NATOM, 3, 3)
          - ratios : (NATOM)
          - widths : (NATOM)
    """

    assert isinstance(molecules, list)
    molecule_list = [util.qcel_to_monomerdata(molecule) for molecule in molecules]

    N = len(molecule_list)

    prds = []

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
        
        atom_model = load_atom_model(modelpath, n_pad)

        prd = predict_monomer_multipoles(atom_model, Rti, Zti, aQti)

        charges = np.array(prd[0][0,:n])

        dipoles = np.array(prd[1][0,:n])

        quadrupoles_ii = prd[2][0,:n]
        quadrupoles_ij = prd[3][0,:n]
        quadrupoles = np.zeros((n, 3, 3))

        quadrupoles[:,0,0] = quadrupoles_ii[:,0] # xx
        quadrupoles[:,1,1] = quadrupoles_ii[:,1] # yy
        quadrupoles[:,2,2] = quadrupoles_ii[:,2] # zz
        quadrupoles[:,0,1] = quadrupoles_ij[:,0] # xy
        quadrupoles[:,0,2] = quadrupoles_ij[:,1] # xz
        quadrupoles[:,1,2] = quadrupoles_ij[:,2] # yz
        quadrupoles[:,1,0] = quadrupoles[:,0,1] # yx
        quadrupoles[:,2,0] = quadrupoles[:,0,2] # zx
        quadrupoles[:,2,1] = quadrupoles[:,1,2] # zy

        ratios = np.array(prd[4][0,:n])

        widths = np.array(prd[5][0,:n])

        print(charges.shape, dipoles.shape, quadrupoles.shape, ratios.shape, widths.shape)

        prd = { "charges" : charges,
                "dipoles" : dipoles,
                "quadrupoles" : quadrupoles,
                "ratios" : ratios,
                "widths" : widths
               }

        prds.append(prd)

    return prds



