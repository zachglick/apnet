import os
import numpy as np
import pandas as pd
import time
import scipy
from scipy.spatial import distance_matrix
import tensorflow as tf
import features
floattype = 'float64'
tf.keras.backend.set_floatx(floattype)

z_to_ind = {
    1  : 0,
    6  : 1,
    7  : 2,
    8  : 3,
    9  : 4,
    11 : 5,
    15 : 6,
    16 : 7,
    17 : 8,
    35 : 9,
}


def pad_molecules(dimer, pad_dim):
    N = len(dimer)
    RA = [dimer[i][0] for i in range(N)]
    RB = [dimer[i][1] for i in range(N)]
    ZA = [dimer[i][2] for i in range(N)]
    ZB = [dimer[i][3] for i in range(N)]
    QA = [dimer[i][4] for i in range(N)]
    QB = [dimer[i][5] for i in range(N)]

    RA_pad = np.zeros((N, pad_dim, 3))
    RB_pad = np.zeros((N, pad_dim, 3))
    ZA_pad = np.zeros((N, pad_dim))
    ZB_pad = np.zeros((N, pad_dim))
    QA_pad = np.zeros((N, pad_dim, 10))
    QB_pad = np.zeros((N, pad_dim, 10))

    for i in range(N):
        natomA = len(ZA[i])
        natomB = len(ZB[i])
        #print(ZA[i])
        #print(RA[i].shape)

        assert natomA == len(RA[i])
        assert natomB == len(RB[i])

        #print(RA_pad[i].shape)
        #print(RA[i].shape)
        RA_pad[i,:natomA,:] = RA[i]
        RB_pad[i,:natomB,:] = RB[i]

        ZA_pad[i,:natomA] = ZA[i]
        ZB_pad[i,:natomB] = ZB[i]

        #print(QA_pad[i].shape)
        #print(QA[i].shape)
        #print(QA[i])
        #print(natomA)
        #exit()
        QA[i] = QA[i][:natomA][:]
        QB[i] = QB[i][:natomB][:]
        QA_pad[i,:natomA,:] = QA[i]
        QB_pad[i,:natomB,:] = QB[i]

    return RA_pad, RB_pad, ZA_pad, ZB_pad, QA_pad, QB_pad

def mse_mp(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=[-1,-2])

def mae_mp(y_true, y_pred):
    if y_pred.shape[-1] == 3:
        return K.mean(K.abs(y_pred - y_true), axis=[-1,-2])
    else:
        return K.mean(K.abs(y_pred - y_true), axis=-1)

class RBFLayer(tf.keras.layers.Layer):
    def __init__(self, mus, etas, **kwargs):
        mus = np.array(mus)
        etas = np.array(etas)
        out_dim = mus.shape[0]
        self.out_dim = out_dim
        self.mus = mus
        self.etas = etas
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mus_w = self.add_weight(name='mus', 
                                     shape=(self.out_dim,),
                                     initializer='uniform',
                                     trainable=True)        
        self.etas_w = self.add_weight(name='etas', 
                                     shape=(self.out_dim,),
                                     initializer='uniform',
                                     trainable=True)        
        self.mus_w.assign(self.mus.reshape(-1))
        self.etas_w.assign(self.etas.reshape(-1))

        self.built = True
        super(RBFLayer, self).build(input_shape)  # Be sure to call this at the end

    def get_config(self):
        config = super(RBFLayer, self).get_config()
        config.update({'mus'  : self.mus,
                       'etas' : self.etas})
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[:-1], self.out_dim)

    def call(self, x, mask=None):
        x_ = tf.tile(x, [1,1,1,self.out_dim])
        x_ = tf.math.subtract(x, self.mus_w)
        x_ = tf.math.square(x_)
        x_ = tf.math.multiply(x_, self.etas_w)
        x_ = tf.math.exp(x_)

        return x_

def print_out_errors(epoch, err, symbol=''):
    """ err is a numpy array w/ dims [NMOL] """
    print((f'{epoch:4d}  ||  '
           f'Range: [{np.min(err):7.3f},{np.max(err):6.3f}]   '
           f'ME:{np.average(err):6.3f}   '
           f'RMSE:{np.sqrt(np.average(np.square(err))):6.3f}   '
           f'MAE:{np.average(np.absolute(err)):6.3f} {symbol}'))

def print_out_errors_comp(err, symbols='    '):
    """ err is a numpy array w/ dims [NMOL x 4] """
    err_tot = np.sum(err, axis=1)
    print((f' Total || '
           f'Range: [{np.min(err_tot):8.3f},{np.max(err_tot):7.3f}]   '
           f'ME:{np.average(err_tot):6.3f}   '
           f'RMSE:{np.sqrt(np.average(np.square(err_tot))):6.3f}   '
           f'MAE:{np.average(np.absolute(err_tot)):6.3f} ||'))
    for ci, cname in enumerate(['Elst', 'Exch', 'Ind ', 'Disp']):
        err_comp = err[:,ci]
        print((f' {cname}  || '
               f'Range: [{np.min(err_comp):8.3f},{np.max(err_comp):7.3f}]   '
               f'ME:{np.average(err_comp):6.3f}   '
               f'RMSE:{np.sqrt(np.average(np.square(err_comp))):6.3f}   '
               f'MAE:{np.average(np.absolute(err_comp)):6.3f} || {symbols[ci]}'))

def int_to_onehot(arr):
    """ arrs is a numpy array of integers w/ dims [NATOM]"""
    assert len(arr.shape) == 1
    arr2 = np.zeros((arr.shape[0], len(z_to_ind)), dtype=np.int)
    for i, z in enumerate(arr):
        if z > 0:
            arr2[i, z_to_ind[z]] = 1
    return arr2

def make_batches(dimer_list, label_list, batch_size=8, order=None):
    dimer_batches = []
    label_batches = []
    N = len(dimer_list)
    for i_start in range(0, N, batch_size):
        i_end = min(i_start + batch_size, N)
        if order is not None:
            # can't slice list w/ arb inds
            #dimer_batch = dimer_list[order[i_start:i_end]]
            #label_batch = label_list[order[i_start:i_end]]
            dimer_batch = [dimer_list[i] for i in order[i_start:i_end]]
            label_batch = [label_list[i] for i in order[i_start:i_end]]
        else:
            dimer_batch = dimer_list[i_start:i_end]
            label_batch = label_list[i_start:i_end]
        dimer_batches.append(dimer_batch)
        label_batches.append(label_batch)
    return dimer_batches, label_batches



def inflate(GA, GB):
    """ GA is the ACSFs of all monomer A atoms with dimensions [NATOMA x NMU x NZ]
        GB is the ACSFs of all monomer B atoms with dimensions [NATOMB x NMU x NZ]
        This function tiles GA and GB so that the first index is a pair of atoms
        Returns GA_ and GB_ both with dimensions [(NATOMA * NATOMB) x NMU x NZ]
        10/13/2020: Updated to also inflate multipoles, dimensions [NATOMA x 10]
     """
    nA, nB = GA.shape[0], GB.shape[0]
    if len(GA.shape) == 3:
        GA_ = np.expand_dims(GA, 1)
        GA_ = np.tile(GA_, (1,nB,1,1))
        GA_ = GA_.reshape(GA_.shape[0] * GA_.shape[1], GA_.shape[2], GA_.shape[3])
        GB_ = np.expand_dims(GB, 1)
        GB_ = np.tile(GB_, (1,nA,1,1))
        GB_ = np.transpose(GB_, (1,0,2,3))
        GB_ = GB_.reshape(GB_.shape[0] * GB_.shape[1], GB_.shape[2], GB_.shape[3])
        return GA_, GB_
    elif len(GA.shape) == 2:
        GA_ = np.expand_dims(GA, 1)
        GA_ = np.tile(GA_, (1,nB,1))
        GA_ = GA_.reshape(GA_.shape[0] * GA_.shape[1], GA_.shape[2])
        GB_ = np.expand_dims(GB, 1)
        GB_ = np.tile(GB_, (1,nA,1))
        GB_ = np.transpose(GB_, (1,0,2))
        GB_ = GB_.reshape(GB_.shape[0] * GB_.shape[1], GB_.shape[2])
        return GA_, GB_


def get_dimers(dataset):
    """
    Get molecular dimer (atoms and coordinates) and SAPT0 labels for a specified dataset

    Args:
        dataset: string corresponding to name of dataset

    Returns tuple of 
    Each element of the tuple is a 
    """

    # load dimer data
    if not os.path.isfile(f'data/{dataset}.pkl'):
       raise Exception(f'No dataset found at data/{dataset}.pkl')
    df = pd.read_pickle(f'data/{dataset}.pkl')

    # extract atom types and atomic coordinates
    ZA = df['ZA'].tolist()
    ZB = df['ZB'].tolist()
    RA = df['RA'].tolist()
    RB = df['RB'].tolist()
    QA = df['multipoles_A'].tolist()
    QB = df['multipoles_B'].tolist()
    #print(QA)
    #print(QA[0].shape)
    #exit()

    dimer = list(zip(RA, RB, ZA, ZB, QA, QB))

    # extract interaction energy label (if specified for the datset)
    try:
        sapt = df[['Elst_aug', 'Exch_aug', 'Ind_aug', 'Disp_aug']].to_numpy()
    except:
        sapt = None

    return dimer, sapt

def make_features(RA, RB, ZA, ZB, QA, QB, MTP, ACSF_nmu=43, APSF_nmu=21, ACSF_eta=100, APSF_eta=25, elst_cutoff=5.0):

    nA = RA.shape[0]
    nB = RB.shape[0]
                                                        
    GA, GB, IA, IB = features.calculate_dimer(RA, RB, ZA, ZB)
    GA, GB = inflate(GA, GB)
    QA, QB = inflate(QA, QB)

    # append 1/D and cutoff to D
    RAB = distance_matrix(RA, RB)
    mask = (RAB <= elst_cutoff).astype(np.float64)
    cutoff = 0.5 * (np.cos(np.pi * RAB / elst_cutoff) + 1) * mask
    RAB = np.stack([RAB, 1.0 / RAB, cutoff], axis=-1)

    # append onehot(Z) to Z
    ZA = np.concatenate([ZA.reshape(-1,1), int_to_onehot(ZA)], axis=1)
    ZB = np.concatenate([ZB.reshape(-1,1), int_to_onehot(ZB)], axis=1)

    # tile ZA by atoms in monomer B and vice versa
    ZA = np.expand_dims(ZA, axis=1)
    ZA = np.tile(ZA, (1, nB, 1))
    ZB = np.expand_dims(ZB, axis=0)
    ZB = np.tile(ZB, (nA,1,1))


    #ZA = ZA.astype(float)
    #ZB = ZA.astype(float)

    # flatten the NA, NB indices
    ZA = ZA.reshape((-1,) + ZA.shape[2:]) 
    ZB = ZB.reshape((-1,) + ZB.shape[2:]) 
    RAB = RAB.reshape((-1,) + RAB.shape[2:])
    IA = IA.reshape((-1,) + IA.shape[2:])
    IB = IB.reshape((-1,) + IB.shape[2:])
    MTP = np.expand_dims(MTP.reshape((-1,)), axis=-1)
 
    # APSF is already made per atom pair 
    # We won't tile ACSFs (which are atomic) into atom pairs b/c memory, do it at runtime instead

    # these are the final shapes:
    # ZA[i]  shape: NA * NB x (NZ + 1)
    # ZB[i]  shape: NA * NB x (NZ + 1)
    # GA[i]  shape: NA x NMU1 x NZ
    # GB[i]  shape: NB x NMU1 x NZ
    # IA[i]  shape: NA * NB x NMU2 x NZ
    # IB[i]  shape: NA * NB x NMU2 x NZ
    # RAB[i] shape: NA * NB x 3
    # y[i]   scalar

    return (ZA, ZB, RAB, GA, GB, IA, IB, QA, QB, MTP)


def make_model(nZ, ACSF_nmu=43, APSF_nmu=21):
    """
    Returns a keras model for atomic pairwise intermolecular energy predictions
    """

    # These three parameters could be experimented with in the future
    # Preliminary tests suggest they aren't that important
    APSF_nodes = 50
    ACSF_nodes = 100
    dense_nodes = 128

    # encoded atomic numbers
    input_layerZA = tf.keras.Input(shape=(nZ+1,), dtype='float64')
    input_layerZB = tf.keras.Input(shape=(nZ+1,), dtype='float64')

    # atom centered symmetry functions
    input_layerGA = tf.keras.Input(shape=(ACSF_nmu,nZ), dtype='float64')
    input_layerGB = tf.keras.Input(shape=(ACSF_nmu,nZ), dtype='float64')

    # atom pair symmetry functions
    input_layerIA = tf.keras.Input(shape=(APSF_nmu,nZ), dtype='float64')
    input_layerIB = tf.keras.Input(shape=(APSF_nmu,nZ), dtype='float64')

    # multipoles
    input_layerQA = tf.keras.Input(shape=(10), dtype='float64')
    input_layerQB = tf.keras.Input(shape=(10), dtype='float64')

    # multipole electrostatics
    input_layerMTP = tf.keras.Input(shape=(1), dtype='float64')

    # interatomic distance in angstrom
    # r, 1/r, and the cutoff function are all passed in, which is redundant but simplifies the code
    input_layerR = tf.keras.Input(shape=(3,), dtype='float64')

    output_layers = []

    for component_ind, component_name in enumerate(['elst', 'exch', 'ind', 'disp']):
        # flatten the symmetry functions
        GA = tf.keras.layers.Flatten()(input_layerGA)
        GB = tf.keras.layers.Flatten()(input_layerGB)
        IA = tf.keras.layers.Flatten()(input_layerIA)
        IB = tf.keras.layers.Flatten()(input_layerIB)

        # encode the concatenation of the element and ACSF into a smaller fixed-length vector
        dense_r = tf.keras.layers.Dense(ACSF_nodes, activation='relu', name=f'{component_name}_dense_r')
        GA = tf.keras.layers.Concatenate()([input_layerZA, GA])
        GA = dense_r(GA)
        GB = tf.keras.layers.Concatenate()([input_layerZB, GB])
        GB = dense_r(GB)

        # encode the concatenation of the element and APSF into a smaller fixed-length vector
        dense_i = tf.keras.layers.Dense(APSF_nodes, activation='relu', name=f'{component_name}_dense_i')
        IA = tf.keras.layers.Concatenate()([input_layerZA, IA])
        IA = dense_i(IA)
        IB = tf.keras.layers.Concatenate()([input_layerZB, IB])
        IB = dense_i(IB)

        # concatenate the atom centered and atom pair symmetry functions
        GA = tf.keras.layers.Concatenate()([GA, IA])
        GB = tf.keras.layers.Concatenate()([GB, IB])

        # concatenate with atom type and distance
        # this is the final input into the feed-forward NN
        AB_ = tf.keras.layers.Concatenate()([input_layerZA, input_layerZB, input_layerR, GA, GB, input_layerMTP])
        BA_ = tf.keras.layers.Concatenate()([input_layerZB, input_layerZA, input_layerR, GB, GA, input_layerMTP])

        # simple feed-forward NN with three dense layers
        dense_1 = tf.keras.layers.Dense(dense_nodes, activation='relu', name=f'{component_name}_dense_1')
        dense_2 = tf.keras.layers.Dense(dense_nodes, activation='relu', name=f'{component_name}_dense_2')
        dense_3 = tf.keras.layers.Dense(dense_nodes, activation='relu', name=f'{component_name}_dense_3')
        linear = tf.keras.layers.Dense(1, activation='linear', name=f'{component_name}_linear', use_bias=False)

        AB_ = dense_1(AB_)
        AB_ = dense_2(AB_)
        AB_ = dense_3(AB_)
        AB_ = linear(AB_)

        BA_ = dense_1(BA_)
        BA_ = dense_2(BA_)
        BA_ = dense_3(BA_)
        BA_ = linear(BA_)

        # symmetrize with respect to A, B
        output_layer = tf.keras.layers.add([AB_, BA_])

        # if electrostatics, introduce switcthing function between scaled MTP and pure MTP
        if component_name == 'elst':
            # choose the following for the distance-scaled energy prediction model
            #output_layer = tf.keras.layers.multiply([output_layer, input_layerR[:,1]])

            # choose the following for the old additive implementation
            #output_layer = tf.keras.layers.multiply([output_layer, input_layerR[:,1]])
            #output_layer = tf.keras.layers.add([output_layer, input_layerMTP])
            
            # choose the following for additive + cutoff implementation (best?):
            cutoff_fn = tf.expand_dims(input_layerR[:,2], -1)
            output_layer = cutoff_fn * output_layer + input_layerMTP

            # choose the following for NN-scaled + cutoff implementation:
            #cutoff_fn = tf.expand_dims(input_layerR[:,2], -1)
            #output_layer = cutoff_fn * output_layer * input_layerMTP + (1.0 - cutoff_fn) * input_layerMTP
            
            #ablation: simply choose the MTP
            #output_layer = input_layerMTP

        # else, simply normalize output by 1/r
        else:
            output_layer = tf.keras.layers.multiply([output_layer, input_layerR[:,1]])
        
        output_layers.append(output_layer)

    output_layer = tf.keras.layers.Concatenate()(output_layers)
    model = tf.keras.Model(inputs=[input_layerZA,
                                   input_layerZB,
                                   input_layerR,
                                   input_layerGA,
                                   input_layerGB, 
                                   input_layerIA, 
                                   input_layerIB,
                                   input_layerQA,
                                   input_layerQB,
                                   input_layerMTP], outputs=output_layer)

    return model


@tf.function(experimental_relax_shapes=True)
def predict_single(model, feat):
    return model(feat)


@tf.function(experimental_relax_shapes=True)
def train_batch(model, optimizer, feats, labels):
    """
    Train the model on a single molecule (batch size == 1).

    Args:
        model: keras model
        optimizer: keras optimizer
        feats: 
        labels:

    Return model error on this molecule (kcal/mol)
    """
    with tf.GradientTape() as tape:
        preds = tf.convert_to_tensor([tf.math.reduce_sum(model(feat), axis=0) for feat in feats])
        err = preds - labels
        err2 = tf.math.square(err)
        loss = tf.math.reduce_mean(err2)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return err

@tf.function(experimental_relax_shapes=True)
def train_single(model, optimizer, feat, label):
    """
    Train the model on a single molecule (batch size == 1).

    Args:
        model: keras model
        optimizer: keras optimizer
        XA: XA
        XB: XB
        XAB: XAB
        GA: GA
        GB: GB
        IA: IA
        IB: IB
        y_label: y_label

    Return model error on this molecule (kcal/mol)
    """
    with tf.GradientTape() as tape:
        pred = tf.math.reduce_sum(model(feat), axis=0)
        err = pred - label
        err2 = tf.math.square(err)
        loss = tf.math.reduce_sum(err2)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return err
