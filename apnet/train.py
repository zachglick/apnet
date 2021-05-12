import os 
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx('float64')

from apnet import util
from apnet import models

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

