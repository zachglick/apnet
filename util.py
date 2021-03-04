import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

import tensorflow as tf
from tensorflow.keras.layers import Add, Concatenate, Dense, Embedding, Flatten, Input, InputLayer, Lambda, Layer, Reshape, Subtract

tf.keras.backend.set_floatx('float64')

import features

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


def int_to_onehot(arr):
    """ arrs is a numpy array of integers w/ dims [NATOM]"""
    assert len(arr.shape) == 1
    arr2 = np.zeros((arr.shape[0], len(z_to_ind)), dtype=np.int)
    for i, z in enumerate(arr):
        if z > 0:
            arr2[i, z_to_ind[z]] = 1
    return arr2

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





class MLP_layer(tf.keras.layers.Layer):
    def __init__(self, nodes, out_dim=1, activation='relu', **kwargs):
        self.nodes = nodes
        self.layer_set = []
        self.out_dim = out_dim
        self.activation = activation
        for num in nodes:
            self.layer_set.append(Dense(num, activation=activation))
        self.layer_set.append(Dense(out_dim, activation=None))
        super(MLP_layer, self).__init__(**kwargs)

    def get_config(self):
        config = super(MLP_layer, self).get_config()
        config.update({
            'nodes': self.nodes,
            'layer_set': self.layer_set,
            'out_dim': self.out_dim,
            'activation': self.activation,
        })
        return config

    @tf.function(experimental_relax_shapes=True)
    def call(self, x):
        for layer in self.layer_set:
            x = layer(x)
        return x

class EPN_layer(tf.keras.layers.Layer):
    """Special 'Electron Passing Network,' which retains conservation of electrons but allows non-local passing"""

    def __init__(self, pass_fn=MLP_layer, T=3, **kwargs):
        self.pass_fns = []
        for t in range(T):
            self.pass_fns.append(pass_fn([32,32]))
        self.T = T
        super(EPN_layer, self).__init__(**kwargs)

    def get_config(self):
        config = super(EPN_layer, self).get_config()
        config.update({
            'pass_fns': self.pass_fns,
            'T': self.T,
        })
        return config

    @tf.function(experimental_relax_shapes=True)
    def call(self, h, e, q, mask):
        tol = tf.constant(1e-8, dtype=tf.float64)
        clip = tf.clip_by_value(e, clip_value_min=tol, clip_value_max=1e5)
        largest_e = tf.reduce_max(clip, axis=-1)
        is_near = tf.math.not_equal(largest_e, tol)
        is_near = tf.cast(is_near, dtype=tf.float64)

        natom = e.shape[1]
        mask = tf.cast(mask, dtype=tf.float64)
        mask_r = tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, natom])
        mask_c = tf.transpose(mask_r, [0, 2, 1])
        exp_mask = mask_r * mask_c
        for t in range(self.T):
            self.pass_fn = self.pass_fns[t]
            inp_atom_i = tf.concat([h, q], axis=-1)  # nmolec x natom x 9+32+1

            inp_i = tf.tile(tf.expand_dims(inp_atom_i, axis=2), [1, 1, natom, 1]) # nmolec x natom x natom x 9+32+1
            inp_j = tf.transpose(inp_i, [0, 2, 1, 3]) #nmolec x natom x natom x 9+32+1
            
            inp_ij_N = tf.concat([inp_i, inp_j, e], axis=-1) #nmolec x natom x natom x 9*2 + 32*2 + 1*2 + 32
            inp_ij_T = tf.concat([inp_j, inp_i, e], axis=-1) #nmolec x natom x natom x 9*2 + 32*2 + 1*2 + 32

            flat_inp_ij = tf.reshape(inp_ij_N, [-1, inp_ij_N.shape[-1]])
            flat_inp_ji = tf.reshape(inp_ij_T, [-1, inp_ij_T.shape[-1]])
            elec_ij_flat = self.pass_fn(flat_inp_ij)
            elec_ji_flat = self.pass_fn(flat_inp_ji)

            elec_ij = tf.reshape(elec_ij_flat, [-1, natom, natom])
            elec_ji = tf.reshape(elec_ji_flat, [-1, natom, natom])

            antisym_pass = 0.5 * (elec_ij - elec_ji) * exp_mask * is_near

            q += tf.expand_dims(tf.reduce_sum(antisym_pass, axis=2), axis=-1)

        return q

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

def make_atom_model(mus, etas, natom, nelem, nembed, nnodes, nmessage):

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

    epn_model = MLP_layer
    y_ = EPN_layer(epn_model, T=nmessage)(h_list[-1], e_, q_, mask)
    
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
    TQA = df['TQA'].tolist()
    TQB = df['TQB'].tolist()
    QA = df['multipoles_A'].tolist()
    QB = df['multipoles_B'].tolist()

    # number of atoms in the monomers
    nA = [np.sum(za > 0) for za in ZA]
    nB = [np.sum(zb > 0) for zb in ZB]

    # average atomic charge of each monomer
    aQA = [TQA[i] / nA[i] for i in range(len(nA))]
    aQB = [TQB[i] / nB[i] for i in range(len(nB))]

    dimer = list(zip(RA, RB, ZA, ZB, aQA, aQB, QA, QB))

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
    return model(feat, training=False)

