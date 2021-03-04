import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import epnn
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Add, Concatenate, Dense, Embedding, Flatten, Input, InputLayer, Lambda, Layer, Reshape, Subtract

tf.keras.backend.set_floatx('float64')

def get_data(path, pad=40):

    df = pd.read_pickle(path)
    nmol = len(df.index)
    R = df['R'].tolist()
    Z = df['Z'].tolist()
    y = df['cartesian_multipoles'].tolist()
    h = df['valence_charges'].tolist()
    v = df['valence_widths'].tolist()


    #h = [_.reshape(-1,1) * 0.0 for _ in h]
    #v = [_.reshape(-1,1) * 0.0 for _ in v]
    for i in range(nmol):
        y[i] = y[i][:,:10]

    for i in range(nmol):
        ni = R[i].shape[0]
    
        tempR = np.zeros((pad, 3))
        tempR[:ni] = R[i]
        R[i] = tempR
    
        tempZ = np.zeros(pad).astype(int)
        tempZ[:ni] = Z[i]
        Z[i] = tempZ

        #tempy = np.zeros((pad,12))
        tempy = np.zeros((pad,10))
        tempy[:ni] = y[i]
        y[i] = tempy
    
    R = np.array(R)
    Z = np.array(Z)
    y = np.array(y)

    n_atoms = [np.count_nonzero(Z[i]) for i in range(len(Z))]
    n_atoms = np.array(n_atoms)

    trace = np.sum(y[:,:,[4,7,9]], axis=2)
    y[:,:,4] -= trace / 3.0
    y[:,:,7] -= trace / 3.0
    y[:,:,9] -= trace / 3.0
    
    #mol_charge = [np.sum(y[i], axis=1)[-1] for i in range(len(y))]
    #mol_charge2 = [np.sum(y[i], axis=1)[0] for i in range(len(y))]
    mol_charge = np.round(np.sum(np.array(y[:,:,0]), axis=1))
    #print(mol_charge)
    #print(mol_charge[:100])
    #print(mol_charge2)
    #print(np.array(y[:,:,0]).shape)
    #exit()
    avg_charge = mol_charge / n_atoms
    #print(avg_charge[:100])
    #exit()
    sq_mask = np.zeros((R.shape[0], R.shape[1], R.shape[1]))
    for i in range(len(sq_mask)):
        for j in range(n_atoms[i]):
            for k in range(n_atoms[i]):
                sq_mask[i][j][k] = 1
    sq_mask = np.expand_dims(sq_mask, axis=-1)
    q_init = [avg_charge[i] * sq_mask[i] for i in range(len(sq_mask))]
    q_init = np.array(q_init)
    return R, Z, q_init, y

def get_dimer_data(path, pad=40):

    df = pd.read_pickle(path)
    nmol = len(df.index)
    RA = df['RA'].tolist()
    RB = df['RB'].tolist()
    ZA = df['ZA'].tolist()
    ZB = df['ZB'].tolist()
    TQA = np.array(df['TQA'].tolist())
    TQB = np.array(df['TQB'].tolist())

    for i in range(nmol):
        nAi = RA[i].shape[0]
        nBi = RB[i].shape[0]
    
        tempRA = np.zeros((pad, 3))
        tempRA[:nAi] = RA[i]
        RA[i] = tempRA
        tempRB = np.zeros((pad, 3))
        tempRB[:nBi] = RB[i]
        RB[i] = tempRB
    
        tempZA = np.zeros(pad).astype(int)
        tempZA[:nAi] = ZA[i]
        ZA[i] = tempZA
        tempZB = np.zeros(pad).astype(int)
        tempZB[:nBi] = ZB[i]
        ZB[i] = tempZB
    
    RA = np.array(RA)
    ZA = np.array(ZA)
    RB = np.array(RB)
    ZB = np.array(ZB)

    n_atomsA = np.array([np.count_nonzero(ZA[i]) for i in range(len(ZA))])
    n_atomsB = np.array([np.count_nonzero(ZB[i]) for i in range(len(ZB))])

    avg_chargeA = TQA / n_atomsA
    avg_chargeB = TQB / n_atomsB

    return RA, RB, ZA, ZB, make_mask(avg_chargeA, n_atomsA, pad), make_mask(avg_chargeB, n_atomsB, pad)

def make_mask(Q, Natom, Npad=40):

    Nmol = Q.shape[0]
    sq_mask = np.zeros((Nmol, Npad, Npad))
    for i in range(Nmol):
        sq_mask[i,:Natom[i],:Natom[i]] = Q[i]
    sq_mask = np.expand_dims(sq_mask, axis=-1)
    return sq_mask

def mse_mp(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=[-1,-2])

def mae_mp(y_true, y_pred):
    if y_pred.shape[-1] == 3:
        return K.mean(K.abs(y_pred - y_true), axis=[-1,-2])
    else:
        return K.mean(K.abs(y_pred - y_true), axis=-1)

def rotation_matrix(n):
    """ returns n rotation matrices for transformations in 3D cartesian space 
        each rotation matrix is generated uniformly at random
        output is an ndarray of dim n x 3 x 3
    """

    q = np.random.normal(loc=0.0, scale=1.0, size=(4,n))
    q = q / np.linalg.norm(q, axis=0)
    q2 = np.square(q)
    R = np.array([ [q2[0] + q2[1] - q2[2] - q2[3],  2*q[1]*q[2] - 2*q[0]*q[3],      2*q[1]*q[3] + 2*q[0]*q[2]     ], 
                   [2*q[1]*q[2] + 2*q[0]*q[3],      q2[0] - q2[1] + q2[2] - q2[3],  2*q[2]*q[3] - 2*q[0]*q[1]     ], 
                   [2*q[1]*q[3] - 2*q[0]*q[2],      2*q[2]*q[3] + 2*q[0]*q[1],      q2[0] - q2[1] - q2[2] + q2[3] ],])

    return np.transpose(R, [2,0,1])

class RBFLayer(Layer):
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

class RotationGenerator(tf.keras.utils.Sequence):
    """generates rotated molecule data"""

    def __init__(self, R, Z, mol_charge, y, batch_size=32, shuffle=True):
        self.R = R # nmol x natom x 3
        self.Z = Z # nmol x natom
        self.mol_charge = mol_charge
        self.y = y # nmol x natom x 10
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.nmol = R.shape[0]
        self.natom = R.shape[1]
        self.mol_inds = np.arange(self.nmol)

        self.on_epoch_end()

    def __len__(self):
        """number of batches per epoch"""
        return int(np.floor(self.nmol / self.batch_size))

    def __getitem__(self, index):
        """generate one batch of data"""
        ind_start = index * self.batch_size
        ind_end = (index + 1) * self.batch_size
        batch_inds = self.mol_inds[ind_start:ind_end]

        R_batch = self.R[batch_inds]
        Z_batch = self.Z[batch_inds]
        Q_batch = self.mol_charge[batch_inds]

        # monopole
        y_batch = self.y[batch_inds][:,:,0]

        # dipole (mu_x, mu_y, mu_z)
        y_i_batch = self.y[batch_inds][:,:,1:4]

        # quadrupole diagonal (Q_xx, Q_yy, Q_zz)
        y_ii_batch = self.y[batch_inds][:,:,[4,7,9]]

        # quadrupole off-diagonal (Q_xy, Q_xz, Q_yz)
        y_ij_batch = self.y[batch_inds][:,:,[5,6,8]]

        return [R_batch, Z_batch, Q_batch], [y_batch, y_i_batch, y_ii_batch, y_ij_batch]

    def on_epoch_end(self):
        """Updates indexes and perform random rotations after each epoch"""
        if self.shuffle == True:
            np.random.shuffle(self.mol_inds)
        M = rotation_matrix(self.nmol)
        for i in range(self.nmol):
            for j in range(self.natom):

                self.R[i,j] = M[i] @ self.R[i,j]

                mu = self.y[i,j][1:4]
                mu = M[i] @ mu
                self.y[i,j][1:4] = mu

                Q = self.y[i,j][[4,5,6,5,7,8,6,8,9]].reshape(3,3)
                Q = M[i] @ Q @ M[i].T
                self.y[i,j][4:10] = Q.flatten()[[0,1,2,4,5,8]]

def get_model(mus, etas, natom, nelem, nembed, nnodes, nmessage):

    # cartesian coordinate input
    input_R = Input(shape=(natom, 3), ragged=False, dtype=tf.float64, name='input_R')                               # nmol x natom x 3

    # atomic number input
    input_Z = Input(shape=(natom,), ragged=False, dtype=tf.float64, name='input_Z')                                  # nmol x natom

    # atom mask (0.0 or 1.0)
    #mask = tf.clip_by_value(input_Z, 0.0, 1.0, name='mask')                         # nmol x natom
    mask = Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0, name='mask'))(input_Z)

    # calculate interatomic displacement
    dR_ = tf.expand_dims(input_R, 2)                                                # nmol x natom x 1 x 3
    #dR_ = tf.tile(dR_, [1, 1, natom, 1])                                            # nmol x natom x natom x 3
    dR_ = tf.tile(dR_, [1, 1, tf.shape(input_R)[1], 1])                                            # nmol x natom x natom x 3
    dR_ = dR_ - tf.transpose(dR_, [0, 2, 1, 3])                                     # nmol x natom x natom x 3

    # construct an initial guess for q (usually Q distributed equally over all atoms)
    input_Q = Input(shape=(natom, natom, 1), dtype=tf.float64, name='input_Q')           # nmol x natom x natom x 1
    q_ = tf.math.divide_no_nan(tf.math.reduce_sum(input_Q, axis=1), tf.math.count_nonzero(input_Q, axis=1, dtype=tf.float64))    # nmol x natom x 1
    #q_ = tf.math.reduce_sum(input_Q, axis=1) / tf.math.count_nonzero(input_Q, axis=1, dtype=tf.float64)    # nmol x natom x 1

    # we'll need info from ei_ to predict a rank-i cartesian tensor
    e_ = tf.norm(dR_, axis=3, keepdims=True)                                        # nmol x natom x natom x 1
    ei_ = tf.math.divide_no_nan(dR_, e_)                                            # nmol x natom x natom x 3
    eij_ = tf.einsum('manx,many->manxy', dR_, dR_)                                  # nmol x natom x natom x 3 x 3

    # get edges for unique cartesian quadrupoles (and separate ii vs ij)
    exx_, eyy_, ezz_ = eij_[:,:,:,0,0], eij_[:,:,:,1,1], eij_[:,:,:,2,2]
    eii_ = tf.stack([exx_, eyy_, ezz_], axis=3)                                     # nmol x natom x natom x 3
    eii_ = tf.math.divide_no_nan(eii_, e_)                                          # nmol x natom x natom x 3
    exy_, exz_, eyz_ = eij_[:,:,:,0,1], eij_[:,:,:,0,2], eij_[:,:,:,1,2]
    eij_ = tf.stack([exy_, exz_, eyz_], axis=3)                                     # nmol x natom x natom x 3
    eij_ = tf.math.divide_no_nan(eij_, e_)                                          # nmol x natom x natom x 3

    # form edge matrix (an edge is a vector of rbfs)
    e_ = RBFLayer(mus, etas, name='rbf')(e_)                                        # nmol x natom x natom x nmu
    ei_ = tf.einsum('manu,manx->manux', e_, ei_)                                    # nmol x natom x natom x nmu x 3
    eii_ = tf.einsum('manu,manx->manux', e_, eii_)                                  # nmol x natom x natom x nmu x 3
    eij_ = tf.einsum('manu,manx->manux', e_, eij_)                                  # nmol x natom x natom x nmu x 3

    # the initial hidden state of an atom, depends only on atom type
    h0_ = Embedding(nelem, nembed, mask_zero=True, name='embed_Z')(input_Z)         # nmol x natom x nembed

    # tile h0_ to 3 dimensions for convenience later
    h0_temp_ = tf.expand_dims(h0_, axis=2)
    h0_temp_ = tf.tile(h0_temp_, [1,1,3,1])                                         # nmol x natom x 3 x nembed

    # saved hidden states
    h_list    = [h0_] # for predicting monopoles, hirshfeld ratios, and valence widths
    h_i_list  = []    # for predicting dipoles
    h_ii_list = []    # for predicting on-diagonal quadrupoles
    h_ij_list = []    # for predicting off-diagonal quadrupoles

    for i in range(nmessage):

        # get the previous rank-0 hidden state
        h_prev_ = h_list[-1]                                                        # nmol x natom x nembed

        # tile over index 1 (every atom gets a copy)
        h_prev_ = tf.expand_dims(h_prev_, 1)                                        # nmol x 1     x natom x nembed
        h_prev_ = tf.tile(h_prev_, [1, natom, 1, 1])                                # nmol x natom x natom x nembed

        #####################################
        # form the next rank-0 hidden state #
        #####################################

        # contract prev hidden state with rank-0 edge info ("interaction layer")
        h_next_ = tf.einsum('manu,mane->manue', e_, h_prev_)                        # nmol x natom x natom x nmu x nembed
        h_next_ = tf.einsum('manue,ma->manue', h_next_, mask)                       # nmol x natom x natom x nmu x nembed
        h_next_ = tf.einsum('manue,mn->manue', h_next_, mask)                       # nmol x natom x natom x nmu x nembed
        h_next_ = tf.math.reduce_sum(h_next_, axis=2)                               # nmol x natom x         nmu x nembed

        # flatten last two dimensions
        #new_shape = [tf.shape(h_next_)[0], h_next_.shape[1], h_next_.shape[2] * h_next_.shape[3]]
        #h_next_ = tf.reshape(h_next_, new_shape)                                   # nmol x natom x nmu,nembed
        new_shape = [h_next_.shape[1], h_next_.shape[2] * h_next_.shape[3]]
        h_next_ = Reshape(new_shape)(h_next_)                                       # nmol x natom x nmu,nembed

        # send though dense network to get new hidden state ("self interaction")
        h_next_ = Concatenate()([h_next_, h0_])
        h_next_ = Dense(nnodes[0], activation='relu')(h_next_)
        h_next_ = Dense(nnodes[1], activation='relu')(h_next_)
        h_next_ = Dense(nnodes[2], activation='relu')(h_next_)
        h_next_ = Dense(nembed, activation='linear')(h_next_)

        # save this new rank-0 hidden state
        h_list.append(h_next_)

        #################################################
        # form the next rank-1 and rank-2 hidden states #
        #################################################

        # contract prev rank-0 hidden state with rank-1 or rank-2 edge info ("interaction layer")
        h_i_ = tf.einsum('manux,mane->manuex', ei_, h_prev_)                        # nmol x natom x natom x nmu x nembed x 3
        h_i_ = tf.einsum('manuex,ma->manuex', h_i_, mask)                           # nmol x natom x natom x nmu x nembed x 3
        h_i_ = tf.einsum('manuex,mn->manuex', h_i_, mask)                           # nmol x natom x natom x nmu x nembed x 3
        h_i_ = tf.math.reduce_sum(h_i_, axis=2)                                     # nmol x natom x         nmu x nembed x 3

        h_ii_ = tf.einsum('manux,mane->manuex', eii_, h_prev_)                      # nmol x natom x natom x nmu x nembed x 3
        h_ii_ = tf.einsum('manuex,ma->manuex', h_ii_, mask)                         # nmol x natom x natom x nmu x nembed x 3
        h_ii_ = tf.einsum('manuex,mn->manuex', h_ii_, mask)                         # nmol x natom x natom x nmu x nembed x 3
        h_ii_ = tf.math.reduce_sum(h_ii_, axis=2)                                   # nmol x natom x         nmu x nembed x 3

        h_ij_ = tf.einsum('manux,mane->manuex', eij_, h_prev_)                      # nmol x natom x natom x nmu x nembed x 3
        h_ij_ = tf.einsum('manuex,ma->manuex', h_ij_, mask)                         # nmol x natom x natom x nmu x nembed x 3
        h_ij_ = tf.einsum('manuex,mn->manuex', h_ij_, mask)                         # nmol x natom x natom x nmu x nembed x 3
        h_ij_ = tf.math.reduce_sum(h_ij_, axis=2)                                   # nmol x natom x         nmu x nembed x 3

        # flatten embedding dimensions and transpose
        #new_shape = [tf.shape(h_i_)[0], h_i_.shape[1], h_i_.shape[2] * h_i_.shape[3], 3]
        #h_i_ = tf.reshape(h_i_, new_shape)                                         # nmol x natom x nmu,nembed x 3
        new_shape = [h_i_.shape[1], h_i_.shape[2] * h_i_.shape[3], 3]
        h_i_ = Reshape(new_shape)(h_i_)                                             # nmol x natom x nmu,nembed x 3
        h_i_ = tf.transpose(h_i_, [0, 1, 3, 2])                                     # nmol x natom x 3 x nmu,nembed

        #new_shape = [tf.shape(h_ii_)[0], h_ii_.shape[1], h_ii_.shape[2] * h_ii_.shape[3], 3]
        #h_ii_ = tf.reshape(h_ii_, new_shape)                                       # nmol x natom x nmu,nembed x 3
        new_shape = [h_ii_.shape[1], h_ii_.shape[2] * h_ii_.shape[3], 3]
        h_ii_ = Reshape(new_shape)(h_ii_)                                           # nmol x natom x nmu,nembed x 3
        h_ii_ = tf.transpose(h_ii_, [0, 1, 3, 2])                                   # nmol x natom x 3 x nmu,nembed

        #new_shape = [tf.shape(h_ij_)[0], h_ij_.shape[1], h_ij_.shape[2] * h_ij_.shape[3], 3]
        #h_ij_ = tf.reshape(h_ij_, new_shape)                                       # nmol x natom x nmu,nembed x 3
        new_shape = [h_ij_.shape[1], h_ij_.shape[2] * h_ij_.shape[3], 3]
        h_ij_ = Reshape(new_shape)(h_ij_)                                           # nmol x natom x nmu,nembed x 3
        h_ij_ = tf.transpose(h_ij_, [0, 1, 3, 2])                                   # nmol x natom x 3 x nmu,nembed

        # send though dense network to get new hidden state ("self interaction")
        h_i_ = Concatenate()([h_i_, h0_temp_])                                      # nmol x natom x 3 x nmu+1,nembed
        h_i_ = Dense(nnodes[0], activation='relu')(h_i_)                            # nmol x natom x 3 x nnodes
        h_i_ = Dense(nnodes[1], activation='relu')(h_i_)                            # nmol x natom x 3 x nnodes
        h_i_ = Dense(nnodes[2], activation='relu')(h_i_)                            # nmol x natom x 3 x nnodes
        h_i_ = Dense(nembed, activation='linear')(h_i_)                             # nmol x natom x 3 x nembed

        h_ii_ = Concatenate()([h_ii_, h0_temp_])                                    # nmol x natom x 3 x nmu+1,nembed
        h_ii_ = Dense(nnodes[0], activation='relu')(h_ii_)                          # nmol x natom x 3 x nnodes
        h_ii_ = Dense(nnodes[1], activation='relu')(h_ii_)                          # nmol x natom x 3 x nnodes
        h_ii_ = Dense(nnodes[2], activation='relu')(h_ii_)                          # nmol x natom x 3 x nnodes
        h_ii_ = Dense(nembed, activation='linear')(h_ii_)                           # nmol x natom x 3 x nembed

        h_ij_ = Concatenate()([h_ij_, h0_temp_])                                    # nmol x natom x 3 x nmu+1,nembed
        h_ij_ = Dense(nnodes[0], activation='relu')(h_ij_)                          # nmol x natom x 3 x nnodes
        h_ij_ = Dense(nnodes[1], activation='relu')(h_ij_)                          # nmol x natom x 3 x nnodes
        h_ij_ = Dense(nnodes[2], activation='relu')(h_ij_)                          # nmol x natom x 3 x nnodes
        h_ij_ = Dense(nembed, activation='linear')(h_ij_)                           # nmol x natom x 3 x nembed

        # save these new rank-1 and rank-2 hidden states
        h_i_list.append(h_i_)
        h_ii_list.append(h_ii_)
        h_ij_list.append(h_ij_)

    epn_model = epnn.MLP_layer
    y_ = epnn.EPN_layer(epn_model, T=nmessage)(h_list[-1], e_, q_, mask)
    
    new_shape = [tf.shape(y_)[0], y_.shape[1]]
    y_ = tf.reshape(y_, new_shape)
    y_ = tf.einsum('ij,ij->ij', y_, mask)

    y_i_list = [Dense(1, activation='linear')(h_) for h_ in h_i_list]
    new_shape = [tf.shape(y_i_list[0])[0], y_i_list[0].shape[1], 3]
    y_i_list = [tf.reshape(y_i_, new_shape) for y_i_ in y_i_list]
    y_i_list = [tf.einsum('ijk,ij->ijk', y_i_, mask) for y_i_ in y_i_list]
    if len(y_i_list) == 1:
        y_i_ = y_i_list[0]
    else:
        y_i_ = Add()(y_i_list)

    y_ii_list = [Dense(1, activation='linear')(h_) for h_ in h_ii_list]
    new_shape = [tf.shape(y_ii_list[0])[0], y_ii_list[0].shape[1], 3]
    y_ii_list = [tf.reshape(y_ii_, new_shape) for y_ii_ in y_ii_list]
    y_ii_list = [tf.einsum('ijk,ij->ijk', y_ii_, mask) for y_ii_ in y_ii_list]
    if len(y_ii_list) == 1:
        y_ii_ = y_ii_list[0]
    else:
        y_ii_ = Add()(y_ii_list)

    y_ij_list = [Dense(1, activation='linear')(h_) for h_ in h_ij_list]
    new_shape = [tf.shape(y_ij_list[0])[0], y_ij_list[0].shape[1], 3]
    y_ij_list = [tf.reshape(y_ij_, new_shape) for y_ij_ in y_ij_list]
    y_ij_list = [tf.einsum('ijk,ij->ijk', y_ij_, mask) for y_ij_ in y_ij_list]
    if len(y_ij_list) == 1:
        y_ij_ = y_ij_list[0]
    else:
        y_ij_ = Add()(y_ij_list)

    y_h_list = [Dense(1, activation='linear')(h_) for h_ in h_list]
    new_shape = [tf.shape(y_h_list[0])[0], y_h_list[0].shape[1]]
    y_h_list = [tf.reshape(y_h_, new_shape) for y_h_ in y_h_list]
    y_h_list = [tf.einsum('ij,ij->ij', y_h_, mask) for y_h_ in y_h_list]
    if len(y_h_list) == 1:
        y_h_ = y_h_list[0]
    else:
        y_h_ = Add()(y_h_list)

    y_v_list = [Dense(1, activation='linear')(h_) for h_ in h_list]
    new_shape = [tf.shape(y_v_list[0])[0], y_v_list[0].shape[1]]
    y_v_list = [tf.reshape(y_v_, new_shape) for y_v_ in y_v_list]
    y_v_list = [tf.einsum('ij,ij->ij', y_v_, mask) for y_v_ in y_v_list]
    if len(y_v_list) == 1:
        y_v_ = y_v_list[0]
    else:
        y_v_ = Add()(y_v_list)

    # this should be 0.0 for neutral monomers
    molecule_charge = tf.math.reduce_sum(y_, axis=1)

    # average this error (extra predicted charge) over atoms of the monomer
    molecule_charge = tf.math.divide(molecule_charge, tf.math.reduce_sum(mask, axis=1))
    error = tf.einsum('ij,i->ij', mask, molecule_charge) 

    # update predictions: guaranteed neutral monomers 
    #output = Subtract()([y_, error])

    # the trace should be 0.0 for each atomic quadrupole (this is trace / 3.0)
    quadrupole_trace = tf.math.reduce_mean(y_ii_, axis=2, keepdims=True)
    y_ii_ = Subtract()([y_ii_, quadrupole_trace])

    y_ = Lambda(lambda x: x, name='q')(y_)
    y_i_ = Lambda(lambda x: x, name='mu_i')(y_i_)
    y_ii_ = Lambda(lambda x: x, name='Q_ii')(y_ii_)
    y_ij_ = Lambda(lambda x: x, name='Q_ij')(y_ij_)
    y_h_ = Lambda(lambda x: x, name='h')(y_h_)
    y_v_ = Lambda(lambda x: x, name='v')(y_v_)

    #output = [y_, y_i_, y_ii_, y_ij_, y_h_, y_v_]
    output = [y_, y_i_, y_ii_, y_ij_]

    model = tf.keras.Model([input_R, input_Z, input_Q], output)

    return model
