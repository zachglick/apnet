""" Subclasses of the tensorflow.keras.Model class.
    These objects should be hidden from the user
 """

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

import logging
tf.get_logger().setLevel(logging.ERROR)

from apnet.layers import DistanceLayer, FeedForwardLayer

#################

target_dim = 1 # target property dimension
max_Z = 35 # largest atomic number

#################

def get_distances(RA, RB, e_source, e_target):

    RA_source = tf.gather(RA, e_source)
    RB_target = tf.gather(RB, e_target)

    dR_xyz = RB_target - RA_source

    dR = tf.sqrt(tf.nn.relu(tf.reduce_sum(dR_xyz ** 2, -1)))

    return dR, dR_xyz

def get_messages(h0, h, rbf, e_source, e_target):

    nedge = tf.shape(e_source)[0]

    h0_source = tf.gather(h0, e_source)
    h0_target = tf.gather(h0, e_target)
    h_source = tf.gather(h, e_source)
    h_target = tf.gather(h, e_target)

    h_all = tf.concat([h0_source, h0_target, h_source, h_target], axis=-1)

    h_all_dot = tf.einsum('ez,er->ezr', h_all, rbf)
    h_all_dot = tf.reshape(h_all_dot, (nedge, -1))

    return tf.concat([h_all, h_all_dot, rbf], axis=-1)

class KerasAtomModel(tf.keras.Model):

    def __init__(self, n_message=3, n_rbf=8, n_neuron=128, n_embed=8, r_cut=5.0):
        super(KerasAtomModel, self).__init__()

        # network hyperparameters
        self.n_message = n_message
        self.n_rbf = n_rbf
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        self.r_cut = r_cut

        # embed interatomic distances into large orthogonal basis
        self.distance_layer = DistanceLayer(n_rbf, r_cut)

        # embed atom types
        self.embed_layer = tf.keras.layers.Embedding(max_Z+1, n_embed)

        # zero-th order charge guess, based solely on atom type
        self.guess_layer = tf.keras.layers.Embedding(max_Z+1, target_dim)

        # update (or interaction) layers for updating hidden states
        self.charge_update_layers = []
        self.dipole_update_layers = []
        self.qpole1_update_layers = []
        self.qpole2_update_layers = []

        # readout layers for predicting multipoles from hidden states
        self.charge_readout_layers = []
        self.dipole_readout_layers = []
        self.qpole_readout_layers = []

        # most layers are feed-forward dense nets with a tapered architecture
        layer_nodes_hidden = [n_neuron * 2, n_neuron, n_neuron // 2, n_embed]
        layer_nodes_readout = [n_neuron * 2, n_neuron, n_neuron // 2, target_dim]
        layer_activations = ["relu", "relu", "relu", "linear"]

        for i in range(n_message):
            self.charge_update_layers.append(FeedForwardLayer(layer_nodes_hidden, layer_activations, f'charge_update_{i}'))
            self.dipole_update_layers.append(FeedForwardLayer(layer_nodes_hidden, layer_activations, f'dipole_update_{i}'))
            self.qpole1_update_layers.append(FeedForwardLayer(layer_nodes_hidden, layer_activations, f'qpole1_update_{i}'))
            self.qpole2_update_layers.append(FeedForwardLayer(layer_nodes_hidden, layer_activations, f'qpole2_update_{i}'))

            self.charge_readout_layers.append(FeedForwardLayer(layer_nodes_readout, layer_activations, f'charge_readout_{i}'))
            self.dipole_readout_layers.append(tf.keras.layers.Dense(1, activation='linear', name=f'dipole_readout_{i}'))
            self.qpole_readout_layers.append(tf.keras.layers.Dense(1, activation='linear', name=f'qpole_readout_{i}'))

    def call(self, inputs):

        ########################
        ### unpack the input ###
        ########################

        Z = inputs['Z']
        R = inputs['R']
        e_source = inputs['e_source']
        e_target = inputs['e_target']
        molecule_ind = inputs['molecule_ind']
        total_charge = inputs['total_charge']

        natom = tf.shape(Z)[0]
        natom_per_mol = tf.math.segment_sum(tf.ones([natom], tf.int32), molecule_ind)

        # [edges]
        dR, dR_xyz = get_distances(R, R, e_source, e_target)

        # [edges x 3]
        dr_unit = dR_xyz / tf.expand_dims(dR, 1)

        # [edges x n_rbf]
        rbf = self.distance_layer(dR)

        #######################
        ### message passing ###
        #######################

        h_list = [tf.keras.layers.Flatten()(self.embed_layer(Z))]
        
        charge = self.guess_layer(Z) # zero-order scalar properties by atom type
        dipole = tf.zeros([natom, 3], tf.float32) # zero-order atomic dipole is 0
        qpole = tf.zeros([natom, 3, 3], tf.float32) # zero-order atomic quadrupole is 0

        for i in range(self.n_message):

            #####################
            ### charge update ###
            #####################

            # [edges x message_embedding_dim]
            m_ij = get_messages(h_list[0], h_list[-1], rbf, e_source, e_target)

            # [atoms x message_embedding_dim]
            m_i = tf.math.unsorted_segment_sum(m_ij, e_source, natom)

            # [atomx x hidden_dim]
            h_next = self.charge_update_layers[i](m_i)
            h_list.append(h_next)
            charge += self.charge_readout_layers[i](h_list[i+1])

            #####################
            ### dipole update ###
            #####################

            m_ij_dipole = self.dipole_update_layers[i](m_ij) # [e x 8]
            m_ij_dipole = tf.einsum('ex,em->exm', dr_unit, m_ij_dipole) # [e x 3 x 8]
            m_i_dipole = tf.math.unsorted_segment_sum(m_ij_dipole, e_source, natom) # [a x 3 x 8]

            d_dipole = self.dipole_readout_layers[i](m_i_dipole) # [a x 3 x 1]
            d_dipole = tf.reshape(d_dipole, [natom, 3]) # [a x 3]
            dipole += d_dipole

            #########################
            ### quadrupole update ###
            #########################

            m_ij_qpole1 = self.qpole1_update_layers[i](m_ij) # [e x 8]
            m_ij_qpole1 = tf.einsum('ex,em->exm', dr_unit, m_ij_qpole1) # [e x 3 x 8]
            m_i_qpole1 = tf.math.unsorted_segment_sum(m_ij_qpole1, e_source, natom) # [a x 3 x 8]

            m_ij_qpole2 = self.qpole2_update_layers[i](m_ij) # [e x 8]
            m_ij_qpole2 = tf.einsum('ex,em->exm', dr_unit, m_ij_qpole2) # [e x 3 x 8]
            m_i_qpole2 = tf.math.unsorted_segment_sum(m_ij_qpole2, e_source, natom) # [a x 3 x 8]

            # todo: try outer product before sum (or is that equivalent?)
            d_qpole = tf.einsum('axf,ayf->axyf', m_i_qpole1, m_i_qpole2)
            d_qpole = d_qpole + tf.transpose(d_qpole, perm=[0,2,1,3])
            d_qpole = self.qpole_readout_layers[i](d_qpole)
            d_qpole = tf.reshape(d_qpole, [natom, 3, 3])
            qpole += d_qpole


        ####################################
        ### enforce traceless quadrupole ###
        ####################################

        qpole_mask = tf.constant([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=tf.float32)
        trace = tf.math.multiply(qpole, qpole_mask)
        trace = tf.math.reduce_sum(trace, axis=[1,2], keepdims=True) / 3.0
        trace = tf.math.multiply(qpole_mask, trace)
        qpole = qpole - trace

        ###################################
        ### enforce charge conservation ###
        ###################################

        total_charge_pred = tf.math.segment_sum(tf.squeeze(charge), molecule_ind)
        total_charge_pred = tf.squeeze(total_charge_pred)
        total_charge_err = total_charge_pred - tf.cast(total_charge, tf.float32)
        charge_err = tf.repeat(total_charge_err / tf.cast(natom_per_mol, tf.float32), natom_per_mol)
        charge_err = tf.reshape(charge_err, [-1,1])
        charge = charge - charge_err

        return charge, dipole, qpole, h_list

    def get_config(self):

        return {
            "n_message" : self.n_message,
            "n_rbf" : self.n_rbf,
            "n_neuron" : self.n_neuron,
            "n_embed" : self.n_embed,
            "r_cut" : self.r_cut,
        }

    @classmethod
    def from_config(cls, config):
        return cls(n_message=config["n_message"],
                   n_rbf=config["n_rbf"],
                   n_neuron=config["n_neuron"],
                   n_embed=config["n_embed"],
                   r_cut=config["r_cut"])

if __name__ == "__main__":

    model = KerasAtomModel()
    print(model.r_cut)

