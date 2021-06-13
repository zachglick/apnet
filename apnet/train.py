import os 
import numpy as np
import math
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx('float64')

from apnet import util
from apnet import models
from apnet import multipoles

def train_multipole_model(molecules_t, multipoles_t, molecules_v, multipoles_v, modelpath):
    """Train a model to predict atom-centered multipoles

    Train a message-passing neural network to predict atomic charges, dipoles, and quadrupoles.
    Requires a dataset of atomic multipoles, obtained from a method like GDMA or MBIS.
    Total molecular charges are conserved, and atomic quadrupoles are traceless.
    This model can handle systems with non-zero total charges.

    Parameters
    ----------
    molecules_t : list of :class:`~qcelemental.models.Molecule`
        Training molecules for the atomic property model
    multipoles_t : list of :class:`~numpy.ndarray`
        Atomic multipoles of molecules_t. Each object in the list is an array with size [natom x 10].
        The ordering convention for the second dimension is [charge, dipole_x, dipole_y, dipole_z,
        quadrupole_xx, quadrupole_xy, quadrupole_xz, quadrupole_yy, quadrupole_yz, quadrupole_zz].
    molecules_v : list of :class:`~qcelemental.models.Molecule`
        Validation molecules for the atomic property model
    multipoles_v : list of :class:`~numpy.ndarray`
        Atomic multipoles of molecules_v. Same convention as multipoles_t
    modelpath : `str`
        Path to save the weights of the trained model to. Must end in ".h5"
    """

    assert isinstance(molecules_t, list)
    assert isinstance(molecules_v, list)


    molecule_list_t = [util.qcel_to_monomerdata(molecule) for molecule in molecules_t]
    molecule_list_v = [util.qcel_to_monomerdata(molecule) for molecule in molecules_v]

    if os.path.isfile(modelpath):
        raise Exception(f"A file already exists at {modelpath}")

    Nt = len(molecule_list_t)
    Nv = len(molecule_list_v)

    pad_dim = np.max([molecule[0].shape[0] for molecule in (molecule_list_t + molecule_list_v)])

    Rt, Zt, q_init_t, MTPt, _, _ = util.padded_monomerdata(pad_dim, molecule_list_t, multipoles_t)
    Rv, Zv, q_init_v, MTPv, _, _ = util.padded_monomerdata(pad_dim, molecule_list_v, multipoles_v)

    nepoch = 200
    nelem = 36
    nembed = 10
    nnodes = [256,128,64]
    nmessage = 3
    nrbf = 43
    napsf = 21
    mus = np.linspace(0.8, 5.0, nrbf)
    etas = np.array([-100.0] * nrbf)
    model = models.make_atom_model(mus, etas, pad_dim, nelem, nembed, nnodes, nmessage, do_properties=False)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=['mse', util.mse_mp, util.mse_mp, util.mse_mp],
                  loss_weights=[1.0, 1.0, 1.0, 1.0],
                  metrics=[util.mae_mp])
    print(model.summary())

    callbacks = [tf.keras.callbacks.ModelCheckpoint(modelpath, save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=(10.0 ** (-1/4)), patience=10, verbose=1, mode='min', min_delta=0, cooldown=0, min_lr=(10.0 ** -5))]


    model.fit(x=util.RotationGenerator(Rt, Zt, q_init_t, MTPt, batch_size=16),
              epochs=nepoch,
              validation_data=([Rv, Zv, q_init_v], [MTPv[:,:,0], MTPv[:,:,1:4], MTPv[:,:,[4,7,9]], MTPv[:,:,[5,6,8]]]),
              callbacks=callbacks,
              verbose=2)


def train_cliff_model(molecules_t, multipoles_t, ratios_t, widths_t, molecules_v, multipoles_v, ratios_v, widths_v, modelpath):
    """Train a model to predict atomic properties used in CLIFF

    Train a message-passing neural network to predict atomic charges, dipoles, and quadrupoles.
    Requires a dataset of atomic multipoles, obtained from a method like GDMA or MBIS.
    Total molecular charges are conserved, and atomic quadrupoles are traceless.
    This model can handle systems with non-zero total charges.

    Parameters
    ----------
    molecules_t : list of :class:`~qcelemental.models.Molecule`
        Training molecules for the atomic property model
    multipoles_t : list of :class:`~numpy.ndarray`
        Atomic multipoles of molecules_t. Each object in the list is an array with size [natom x 10].
        The ordering convention for the second dimension is [charge, dipole_x, dipole_y, dipole_z,
        quadrupole_xx, quadrupole_xy, quadrupole_xz, quadrupole_yy, quadrupole_yz, quadrupole_zz].
    molecules_v : list of :class:`~qcelemental.models.Molecule`
        Validation molecules for the atomic property model
    multipoles_v : list of :class:`~numpy.ndarray`
        Atomic multipoles of molecules_v. Same convention as multipoles_v
    modelpath : `str`
        Path to save the weights of the trained model to. 
    """

    assert isinstance(molecules_t, list)
    assert isinstance(molecules_v, list)

    molecule_list_t = [util.qcel_to_monomerdata(molecule) for molecule in molecules_t]
    molecule_list_v = [util.qcel_to_monomerdata(molecule) for molecule in molecules_v]

    if os.path.isfile(modelpath):
        raise Exception(f"A file already exists at {modelpath}")

    Nt = len(molecule_list_t)
    Nv = len(molecule_list_v)

    pad_dim = np.max([molecule[0].shape[0] for molecule in (molecule_list_t + molecule_list_v)])


    Rt, Zt, q_init_t, MTPt, RATt, WIDt = util.padded_monomerdata(pad_dim, molecule_list_t, multipoles_t, ratios_t, widths_t)
    Rv, Zv, q_init_v, MTPv, RATv, WIDv = util.padded_monomerdata(pad_dim, molecule_list_v, multipoles_v, ratios_v, widths_v)

    nepoch = 200
    nelem = 36
    nembed = 10
    nnodes = [256,128,64]
    nmessage = 3
    nrbf = 43
    napsf = 21
    mus = np.linspace(0.8, 5.0, nrbf)
    etas = np.array([-100.0] * nrbf)
    model = models.make_atom_model(mus, etas, pad_dim, nelem, nembed, nnodes, nmessage, do_properties=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
                  loss=['mse', util.mse_mp, util.mse_mp, util.mse_mp, 'mse', 'mse'],
                  loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                  metrics=[util.mae_mp])
    print(model.summary())

    callbacks = [tf.keras.callbacks.ModelCheckpoint(modelpath, save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=(10.0 ** (-1/5)), patience=8, verbose=1, mode='min', min_delta=0, cooldown=0, min_lr=(10.0 ** -5))]

    model.fit(x=util.RotationGenerator(Rt, Zt, q_init_t, np.concatenate([MTPt, RATt, WIDt], axis=2), batch_size=16),
              epochs=nepoch,
              validation_data=([Rv, Zv, q_init_v], [MTPv[:,:,0], MTPv[:,:,1:4], MTPv[:,:,[4,7,9]], MTPv[:,:,[5,6,8]], RATv, WIDv]),
              callbacks=callbacks,
              verbose=2)

from pathlib import Path

ROOT_DIR = Path(__file__).parent

atom_model_cache = {}
default_atom_modelpaths = [f'{ROOT_DIR}/atom_models/hfadz{i+1}.hdf5' for i in range(3)]

nelem = 36
nembed = 10
nnodes = [256,128,64]
nmessage = 3
nrbf = 43
napsf = 21
mus = np.linspace(0.8, 5.0, nrbf)
etas = np.array([-100.0] * nrbf)

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

def make_batches(dimer_list, label_list, batch_size=8, order=None):
    dimer_batches = []
    label_batches = []
    N = len(dimer_list)
    for i_start in range(0, N, batch_size):
        i_end = min(i_start + batch_size, N)
        if order is not None:
            dimer_batch = [dimer_list[i] for i in order[i_start:i_end]]
            label_batch = [label_list[i] for i in order[i_start:i_end]]
        else:
            dimer_batch = dimer_list[i_start:i_end]
            label_batch = label_list[i_start:i_end]
        dimer_batches.append(dimer_batch)
        label_batches.append(label_batch)
    return dimer_batches, label_batches

@tf.function(experimental_relax_shapes=True)
def predict_monomer_multipoles(model, R, Z, mask):
    return model([R, Z, mask], training=False)

@tf.function(experimental_relax_shapes=True)
def train_batch(model, optimizer, feats, labels):
    """ 
    Train the model on a batch of molecules

    Args:
        model: keras model
        optimizer: keras optimizer
        feats: 
        labels:

    Return model error on this molecule (kcal/mol)
    """
    # todo: check loss
    with tf.GradientTape() as tape:
        preds = tf.convert_to_tensor([tf.math.reduce_sum(model(feat), axis=0) for feat in feats])
        err = preds - labels
        err2 = tf.math.square(err)
        loss = tf.math.reduce_mean(err2)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return err 

@tf.function(experimental_relax_shapes=True)
def predict_single(model, feat):
    return model(feat)

def do_dimer_elst(dimer_list):

    # for now, users can't choose the atom model
    atom_modelpaths = default_atom_modelpaths

    dimer_list_updated = []

    for i, d in enumerate(dimer_list):

        if d is None:
            print('Error!')
            exit()

        nA, nB = len(d[0]), len(d[1])
        nA_pad, nB_pad = 10 * ((nA + 9) // 10), 10 * ((nB + 9) // 10)

        atom_models_A = [load_atom_model(path, nA_pad) for path in atom_modelpaths]
        atom_models_B = [load_atom_model(path, nB_pad) for path in atom_modelpaths]

        RAi = np.zeros((1, nA_pad, 3))
        RBi = np.zeros((1, nB_pad, 3))
        RAi[0,:nA,:] = d[0] 
        RBi[0,:nB,:] = d[1] 

        ZAi = np.zeros((1, nA_pad))
        ZBi = np.zeros((1, nB_pad))
        ZAi[0,:nA] = d[2] 
        ZBi[0,:nB] = d[3] 

        aQAi = np.zeros((1, nA_pad, nA_pad, 1))
        aQBi = np.zeros((1, nB_pad, nB_pad, 1))
        aQAi[0,:nA,:nA,0] = d[4]
        aQBi[0,:nB,:nB,0] = d[5]

        mtpA_prds = []
        mtpB_prds = []
        
        for atom_model in atom_models_A:
            mtpA_prd = predict_monomer_multipoles(atom_model, RAi, ZAi, aQAi)
            mtpA_prd = np.concatenate([np.expand_dims(mtpA_prd[0], axis=-1), mtpA_prd[1], mtpA_prd[2], mtpA_prd[3]], axis=-1)
            mtpA_prds.append(mtpA_prd)

        for atom_model in atom_models_B:
            mtpB_prd = predict_monomer_multipoles(atom_model, RBi, ZBi, aQBi)
            mtpB_prd = np.concatenate([np.expand_dims(mtpB_prd[0], axis=-1), mtpB_prd[1], mtpB_prd[2], mtpB_prd[3]], axis=-1)
            mtpB_prds.append(mtpB_prd)

        mtpA = np.average(mtpA_prds, axis=0)[0,:nA,:]
        mtpB = np.average(mtpB_prds, axis=0)[0,:nB,:]

        elst, pair_mtp = multipoles.eval_dimer(d[0], d[1], d[2], d[3], mtpA, mtpB)
        dimer_list_updated.append((d[0], d[1], d[2], d[3], mtpA, mtpB, pair_mtp))

    return dimer_list_updated


def train_sapt_model(dimers_t, energies_t, dimers_v, energies_v, modelpath, **kwargs):
    """Train a model to predict interaction energies via the SAPT decomposition

    Train an atomic pairwise neural network to predict interaction energies decomposed
    into electrostatics, exchange, induction, and dispersion. Requires a dataset of
    decomposed interaction energies.

    Parameters
    ----------
    dimers_t : list of :class:`~qcelemental.models.Molecule`
        Training dimers for the SAPT interaction energy model.
    energies_t : list of :class:`~numpy.ndarray`
        Interaction energies for dimers_t. Each object in the list is a one-dimensional array of length 4.
        The ordering convention for the array is [electrostatics, exchange, induction, dispersion].
    dimers_v : list of :class:`~qcelemental.models.Molecule`
        Validation dimers for the SAPT interaction energy model.
    energies_v : list of :class:`~numpy.ndarray`
        Interaction energies for dimers_v. Same convention as energies_t.
    modelpath : `str`
        Path to save the weights of the trained model to. Must end in ".h5"
    **kwargs 
        Arbitrary keyword arguments for training. TODO more docs
    """

    assert isinstance(dimers_t, list)
    assert isinstance(dimers_v, list)
    assert modelpath.endswith(".h5")

    print('Validating Dimers...\n')
    dimer_list_t = [util.qcel_to_dimerdata(dimer) for dimer in dimers_t]
    dimer_list_v = [util.qcel_to_dimerdata(dimer) for dimer in dimers_v]

    if os.path.isfile(modelpath):
        raise Exception(f"A file already exists at {modelpath}")

    batch_size = kwargs.pop("batch_size", 8)
    adam_lr = kwargs.pop("adam_lr", 0.001)
    decay_rate = kwargs.pop("decay_rate", 0.2)
    epochs = kwargs.pop("epochs", 200)

    Nt = len(dimer_list_t)
    Nv = len(dimer_list_v)
    Nb = math.ceil(Nt / batch_size)

    # for now, users can't choose the atom model
    atom_modelpaths = default_atom_modelpaths

    print('Calculating Multipoles and Long-Range Electrostatics...\n')
    dimer_list_t = do_dimer_elst(dimer_list_t)
    dimer_list_v = do_dimer_elst(dimer_list_v)

    pair_model = models.make_pair_model(nZ=nembed, ACSF_nmu=nrbf, APSF_nmu=napsf)
    pair_model.save(f'{modelpath}')
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate = adam_lr,
        decay_steps = Nb, 
        decay_rate = decay_rate,
        staircase=True, 
    )   
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    best_mae_v = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
    pair_model_best = models.make_pair_model(nZ=nembed, ACSF_nmu=nrbf, APSF_nmu=napsf)
    pair_model_best.load_weights(modelpath)

    print('Training Atom-Pair Energy Model...\n')

    for epoch in range(epochs):

        #print(f"  EPOCH {epoch+1}")
        print(f"    Epoch {epoch+1:<5d}     Total      Elst       Exch       Ind        Disp")

        epoch_start_time = time.time()

        # shuffle the training dimers every epoch
        inds = np.random.permutation(Nt).astype(int)

        # collect errors on training and validation dimers this epoch
        energy_errs_t, energy_errs_v = [], [] 

        # train on batches of training dimers
        dimer_batches, energy_batches = make_batches(dimer_list_t, energies_t, batch_size, order=inds)
        for dimer_batch, energy_batch in zip(dimer_batches, energy_batches):
            feature_batch = [util.make_features(*d) for d in dimer_batch]
            energy_err_t = train_batch(pair_model, optimizer, feature_batch, energy_batch)
            energy_errs_t.append(energy_err_t)
        energy_errs_t = np.concatenate(energy_errs_t)
        energy_errs_t = np.concatenate([np.sum(energy_errs_t, axis=1, keepdims=True), energy_errs_t], axis=1)
        energy_maes_t = np.average(np.absolute(energy_errs_t), axis=0)
        print(f"    Train MAE: {energy_maes_t[0]:10.3f} {energy_maes_t[1]:10.3f} {energy_maes_t[2]:10.3f} {energy_maes_t[3]:10.3f} {energy_maes_t[4]:10.3f}")

        # infer on validation dimers
        dimer_batches, label_batches = make_batches(dimer_list_v, energies_v, batch_size)
        for dimer_batch, label_batch in zip(dimer_batches, label_batches):
            feature_batch = [util.make_features(*d) for d in dimer_batch]
            for fv, lv in zip(feature_batch, label_batch):
                energy_pred_v = predict_single(pair_model, fv) 
                energy_pred_v = np.sum(energy_pred_v, axis=0)
                energy_errs_v.append(energy_pred_v - lv) 
        energy_errs_v = np.array(energy_errs_v)
        energy_errs_v = np.concatenate([np.sum(energy_errs_v, axis=1, keepdims=True), energy_errs_v], axis=1)
        energy_maes_v = np.average(np.absolute(energy_errs_v), axis=0)
        print(f"    Val   MAE: {energy_maes_v[0]:10.3f} {energy_maes_v[1]:10.3f} {energy_maes_v[2]:10.3f} {energy_maes_v[3]:10.3f} {energy_maes_v[4]:10.3f}")

        # save weights per SAPT component (if validation error improved)
        improved_mae = np.greater(best_mae_v, energy_maes_v)[1:]
        improved_comps = []
        for ci, cname in enumerate(['elst', 'exch', 'ind', 'disp']):
            if improved_mae[ci]:
                best_mae_v[ci+1] = energy_maes_v[ci+1] # +1 because no total
                for layer_index, layer in enumerate(pair_model.layers):
                    if layer.name.startswith(cname):
                        pair_model_best.layers[layer_index].set_weights(layer.get_weights())
                improved_comps.append(cname)

        improved_comps = ",".join(improved_comps)
        if improved_mae.any():
            pair_model_best.save(modelpath)

        epoch_dt = int(time.time() - epoch_start_time)
        print(f"    Elapsed Time: {epoch_dt} sec")
        print(f"    Improved Components: {improved_comps}\n")

@tf.function(experimental_relax_shapes=True)
def transfer_batch(model, optimizer, feats, labels):
    """ 
    Train the model on a batch of molecules

    Args:
        model: keras model
        optimizer: keras optimizer
        feats: 
        labels:

    Return model error on this molecule (kcal/mol)
    """
    # todo: check loss
    with tf.GradientTape() as tape:
        preds = tf.convert_to_tensor([tf.math.reduce_sum(model(feat)) for feat in feats])
        err = preds - labels
        err2 = tf.math.square(err)
        loss = tf.math.reduce_mean(err2)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return err 

def transfer_sapt_model(dimers_t, energies_t, dimers_v, energies_v, modelpath_old, modelpath_new, **kwargs):
    """ Perform transfer learning with an interaction energy model (from the SAPT decomposition to
    some other interaction energy)

    With an atomic-pairwise neural network trained to the SAPT decompostion as a starting point, 
    perform "transfer learning" to some other interaction energy.

    Parameters
    ----------
    dimers_t : list of :class:`~qcelemental.models.Molecule`
        Training dimers for the SAPT interaction energy model.
    energies_t : list of :class:`~numpy.ndarray`
        Interaction energies for dimers_t. Each object in the list is a single scalar, which is the
        interaction energy at some desired level of theory. No SAPT decomposition
    dimers_v : list of :class:`~qcelemental.models.Molecule`
        Validation dimers for the SAPT interaction energy model.
    energies_v : list of :class:`~numpy.ndarray`
        Interaction energies for dimers_v. Same convention as energies_t.
    modelpath_old : `str`
        Path to file of pre-trained energy model weights to be transfer learned with.
    modelpath_new : `str`
        Path to save the weights of the transfer learned model. Must end in ".h5"
    **kwargs 
        Arbitrary keyword arguments for training. TODO more docs
    """

    assert isinstance(dimers_t, list)
    assert isinstance(dimers_v, list)
    assert modelpath_new.endswith(".h5")

    print('Validating Dimers...\n')
    dimer_list_t = [util.qcel_to_dimerdata(dimer) for dimer in dimers_t]
    dimer_list_v = [util.qcel_to_dimerdata(dimer) for dimer in dimers_v]

    if os.path.isfile(modelpath_new):
        raise Exception(f"A file already exists at {modelpath_new}")

    batch_size = kwargs.pop("batch_size", 8)
    adam_lr = kwargs.pop("adam_lr", 0.001)
    decay_rate = kwargs.pop("decay_rate", 0.2)
    epochs = kwargs.pop("epochs", 200)

    print(batch_size, adam_lr, decay_rate, epochs)

    Nt = len(dimer_list_t)
    Nv = len(dimer_list_v)
    Nb = math.ceil(Nt / batch_size)

    # for now, users can't choose the atom model
    atom_modelpaths = default_atom_modelpaths

    print('Calculating Multipoles and Long-Range Electrostatics...\n')
    dimer_list_t = do_dimer_elst(dimer_list_t)
    dimer_list_v = do_dimer_elst(dimer_list_v)

    pair_model = models.make_pair_model(nZ=nembed, ACSF_nmu=nrbf, APSF_nmu=napsf)
    pair_model.load_weights(modelpath_old)
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate = adam_lr,
        decay_steps = Nb, 
        decay_rate = decay_rate,
        staircase=True, 
    )   
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    best_mae_v = np.inf

    for layer_index, layer in enumerate(pair_model.layers):
        if "linear" in layer.name:
            print(f"Leaving layer '{layer.name}' unfrozen for transfer learning")
        else:
            layer.trainable = False

    print('Training Atom-Pair Energy Model...\n')

    print(f"    Pre-Training")

    # collect errors on training and validation dimers this epoch
    energy_errs_t, energy_errs_v = [], [] 

    # train on batches of training dimers
    dimer_batches, energy_batches = make_batches(dimer_list_t, energies_t, batch_size)
    for dimer_batch, energy_batch in zip(dimer_batches, energy_batches):
        feature_batch = [util.make_features(*d) for d in dimer_batch]
        for ft, lt in zip(feature_batch, energy_batch):
            energy_pred_t = predict_single(pair_model, ft) 
            energy_pred_t = np.sum(energy_pred_t)
            energy_errs_t.append(energy_pred_t - lt) 
    energy_errs_t = np.array(energy_errs_t)
    energy_maes_t = np.average(np.absolute(energy_errs_t))
    print(f"    Train MAE: {energy_maes_t:10.3f}")

    # infer on validation dimers
    dimer_batches, energy_batches = make_batches(dimer_list_v, energies_v, batch_size)
    for dimer_batch, energy_batch in zip(dimer_batches, energy_batches):
        feature_batch = [util.make_features(*d) for d in dimer_batch]
        for fv, lv in zip(feature_batch, energy_batch):
            energy_pred_v = predict_single(pair_model, fv) 
            energy_pred_v = np.sum(energy_pred_v)
            energy_errs_v.append(energy_pred_v - lv) 
    energy_errs_v = np.array(energy_errs_v)
    energy_maes_v = np.average(np.absolute(energy_errs_v))
    print(f"    Val   MAE: {energy_maes_v:10.3f}\n")

    best_mae_v = energy_maes_v

    for epoch in range(epochs):

        print(f"    Epoch {epoch+1:<5d}")

        epoch_start_time = time.time()

        # shuffle the training dimers every epoch
        inds = np.random.permutation(Nt).astype(int)

        # collect errors on training and validation dimers this epoch
        energy_errs_t, energy_errs_v = [], [] 

        # train on batches of training dimers
        dimer_batches, energy_batches = make_batches(dimer_list_t, energies_t, batch_size, order=inds)
        for dimer_batch, energy_batch in zip(dimer_batches, energy_batches):
            feature_batch = [util.make_features(*d) for d in dimer_batch]
            energy_err_t = transfer_batch(pair_model, optimizer, feature_batch, energy_batch)
            energy_errs_t.append(energy_err_t)
        energy_errs_t = np.concatenate(energy_errs_t)
        energy_maes_t = np.average(np.absolute(energy_errs_t))
        print(f"    Train MAE: {energy_maes_t:10.3f}")

        # infer on validation dimers
        dimer_batches, energy_batches = make_batches(dimer_list_v, energies_v, batch_size)
        for dimer_batch, energy_batch in zip(dimer_batches, energy_batches):
            feature_batch = [util.make_features(*d) for d in dimer_batch]
            for fv, lv in zip(feature_batch, energy_batch):
                energy_pred_v = predict_single(pair_model, fv) 
                energy_pred_v = np.sum(energy_pred_v)
                energy_errs_v.append(energy_pred_v - lv) 
        energy_errs_v = np.array(energy_errs_v)
        energy_maes_v = np.average(np.absolute(energy_errs_v))
        print(f"    Val   MAE: {energy_maes_v:10.3f}")

        if energy_maes_v < best_mae_v:
            best_mae_v = energy_maes_v
            pair_model.save(modelpath_new)
            print('Saved !!')

        epoch_dt = int(time.time() - epoch_start_time)
        print(f"    Elapsed Time: {epoch_dt} sec\n")
