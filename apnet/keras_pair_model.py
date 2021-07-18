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

def get_pair(hA, hB, rbf, e_source, e_target):

    hA_source = tf.gather(hA, e_source)
    hB_target = tf.gather(hB, e_target)

    # todo: outer product
    return tf.concat([hA_source, hB_target, rbf], axis=-1)

class KerasPairModel(tf.keras.Model):

    def __init__(self, atom_model, n_message=3, n_rbf=8, n_neuron=128, n_embed=8, r_cut_im=8.0):
        super(KerasPairModel, self).__init__()

        # pre-trained atomic model for predicting atomic properties
        self.atom_model = atom_model
        self.atom_model.trainable = False

        # network hyperparameters
        self.n_message = n_message
        self.n_rbf = n_rbf
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        self.r_cut_im = r_cut_im
        #self.r_cut = 5.0

        # embed interatomic distances into large orthogonal basis
        self.distance_layer_im = DistanceLayer(n_rbf, r_cut_im)

        # embed atom types
        self.embed_layer = tf.keras.layers.Embedding(max_Z+1, n_embed)

        ## pre-trained atomic model for predicting atomic properties
        #self.atom_model = keras.models.load_model("/storage/home/hhive1/zglick3/data/test_apnet/atom_models/atom0/")
        #self.atom_model.trainable = False

        # the architecture contains many feed-forward dense nets with a tapered architecture
        layer_nodes_hidden = [n_neuron * 2, n_neuron, n_neuron // 2, n_embed]
        layer_nodes_readout = [n_neuron * 2, n_neuron, n_neuron // 2, 1]
        layer_activations = ["relu", "relu", "relu", "linear"]

        self.readout_layer_elst = FeedForwardLayer(layer_nodes_readout, layer_activations, f'readout_layer_elst')
        self.readout_layer_exch = FeedForwardLayer(layer_nodes_readout, layer_activations, f'readout_layer_exch')
        self.readout_layer_ind = FeedForwardLayer(layer_nodes_readout, layer_activations, f'readout_layer_ind')
        self.readout_layer_disp = FeedForwardLayer(layer_nodes_readout, layer_activations, f'readout_layer_disp')

        # embed distances into large orthogonal basis
        self.distance_layer = DistanceLayer(n_rbf, 5.0)

        self.update_layers = []
        self.readout_layers = []
        self.directional_layers = []
        self.directional_readout_layers = []

        for i in range(self.n_message):

            self.update_layers.append(FeedForwardLayer(layer_nodes_hidden, layer_activations, f'update_layer_{i}'))
            self.readout_layers.append(FeedForwardLayer(layer_nodes_readout, layer_activations, f'readout_layer_{i}'))

            self.directional_layers.append(FeedForwardLayer(layer_nodes_hidden, layer_activations, f'directional_layer_{i}'))
            self.directional_readout_layers.append(tf.keras.layers.Dense(1, activation='linear'))

    def mtp_elst(self, qA, muA, quadA, qB, muB, quadB, e_ABsr_source, e_ABsr_target, dR_ang, dR_xyz_ang):

        dR = dR_ang / 0.529177
        dR_xyz = dR_xyz_ang / 0.529177
        oodR = tf.math.reciprocal(dR)

        delta = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        qA_source = tf.gather(tf.squeeze(qA), e_ABsr_source)
        qB_source = tf.gather(tf.squeeze(qB), e_ABsr_target)

        muA_source = tf.gather(muA, e_ABsr_source)
        muB_source = tf.gather(muB, e_ABsr_target)

        quadA_source = (3.0 / 2.0) * tf.gather(quadA, e_ABsr_source)
        quadB_source = (3.0 / 2.0) * tf.gather(quadB, e_ABsr_target)

        E_qq = tf.einsum("x,x,x->x", qA_source, qB_source, oodR)
        
        T1 = tf.einsum('x,xy->xy', oodR ** 3, -1.0 * dR_xyz)
        qu = tf.einsum('x,xy->xy', qA_source, muB_source) - tf.einsum('x,xy->xy', qB_source, muA_source)
        E_qu = tf.einsum('xy,xy->x', T1, qu)

        T2 = 3 * tf.einsum('xy,xz->xyz', dR_xyz, dR_xyz) - tf.einsum('x,x,yz->xyz', dR, dR, delta)
        T2 = tf.einsum('x,xyz->xyz', oodR ** 5, T2)

        # this is basically zero?
        E_uu = -1.0 * tf.einsum('xy,xz,xyz->x', muA_source, muB_source, T2)

        qA_quadB_source = tf.einsum('x,xyz->xyz', qA_source, quadB_source)
        qB_quadA_source = tf.einsum('x,xyz->xyz', qB_source, quadA_source)
        E_qQ = tf.einsum('xyz,xyz->x', T2, qA_quadB_source + qB_quadA_source) / 3.0

        E_elst =  627.509 * (E_qq + E_qu + E_qQ + E_uu)
        #E_elst =  627.509 * (E_qq + E_qu)
        #E_elst =  627.509 * (E_qq + E_qu + E_uu)

        return E_elst




    def call(self, inputs):

        ########################
        ### unpack the input ###
        ########################

        # monomer atom coordinates and types
        ZA = inputs['ZA']
        RA = inputs['RA']
        ZB = inputs['ZB']
        RB = inputs['RB']

        # short range, intermolecular edges
        e_ABsr_source = inputs['e_ABsr_source']
        e_ABsr_target = inputs['e_ABsr_target']
        dimer_ind = inputs['dimer_ind']

        # long range, intermolecular edges
        e_ABlr_source = inputs['e_ABlr_source']
        e_ABlr_target = inputs['e_ABlr_target']
        dimer_ind_lr = inputs['dimer_ind_lr']

        # intramonomer edges (monomer A)
        e_AA_source = inputs['e_AA_source']
        e_AA_target = inputs['e_AA_target']

        # intramonomer edges (monomer B)
        e_BB_source = inputs['e_BB_source']
        e_BB_target = inputs['e_BB_target']

        # counts
        natomA = tf.shape(ZA)[0]
        natomB = tf.shape(ZB)[0]
        ndimer = tf.shape(inputs['total_charge_A'])[0]
        nedge_sr = tf.shape(e_ABsr_source)[0]
        nedge_lr = tf.shape(e_ABlr_source)[0]

        # interatomic distances
        dR_sr, dR_sr_xyz = get_distances(RA, RB, e_ABsr_source, e_ABsr_target)
        dR_lr, dR_lr_xyz = get_distances(RA, RB, e_ABlr_source, e_ABlr_target)
        dRA, dRA_xyz  = get_distances(RA, RA, e_AA_source, e_AA_target)
        dRB, dRB_xyz  = get_distances(RB, RB, e_BB_source, e_BB_target)

        # interatomic unit vectors
        dR_sr_unit = dR_sr_xyz / tf.expand_dims(dR_sr, 1)
        dRA_unit = dRA_xyz / tf.expand_dims(dRA, 1)
        dRB_unit = dRB_xyz / tf.expand_dims(dRB, 1)

        # distance encodings
        #rbf_sr = self.rbf_layer_im(dR_sr)
        rbf_sr = self.distance_layer_im(dR_sr)
        rbfA = self.distance_layer(dRA)
        rbfB = self.distance_layer(dRB)

        ##########################################################
        ### predict monomer properties w/ pretrained AtomModel ###
        ##########################################################

        inputsA = {
                'Z' : inputs['ZA'],
                'R' : inputs['RA'],
                'e_source' : inputs['e_AA_source'],
                'e_target' : inputs['e_AA_target'],
                'molecule_ind' : inputs['monomerA_ind'],
                'total_charge' : inputs['total_charge_A']
        }

        inputsB = {
                'Z' : inputs['ZB'],
                'R' : inputs['RB'],
                'e_source' : inputs['e_BB_source'],
                'e_target' : inputs['e_BB_target'],
                'molecule_ind' : inputs['monomerB_ind'],
                'total_charge' : inputs['total_charge_B']
        }

        qA, muA, quadA, hlistA = self.atom_model(inputsA)
        qB, muB, quadB, hlistB = self.atom_model(inputsB)

        ################################################################
        ### predict SAPT components via intramonomer message passing ###
        ################################################################
        
        # invariant hidden state lists
        # each list element is [natomA/B x nembed]
        hA_list = [tf.keras.layers.Flatten()(self.embed_layer(ZA))]
        hB_list = [tf.keras.layers.Flatten()(self.embed_layer(ZB))]

        # directional hidden state lists
        # each list element is [natomA/B x 3 x nembed]
        hA_dir_list = []
        hB_dir_list = []

        for i in range(self.n_message):

            # intramonomer messages (from atom a to a' and from b to b')
            # [intrmonomer_edges x message_size]
            mA_ij = get_messages(hA_list[0], hA_list[-1], rbfA, e_AA_source, e_AA_target)
            mB_ij = get_messages(hB_list[0], hB_list[-1], rbfB, e_BB_source, e_BB_target)

            #################
            ### invariant ###
            #################

            # sum each atom's messages
            # [atoms x message_size]
            mA_i = tf.math.unsorted_segment_sum(mA_ij, e_AA_source, natomA)
            mB_i = tf.math.unsorted_segment_sum(mB_ij, e_BB_source, natomB)

            # get the next hidden state of the atom
            # [atomx x hidden_dim]
            hA_next = self.update_layers[i](mA_i)
            hB_next = self.update_layers[i](mB_i)

            hA_list.append(hA_next)
            hB_list.append(hB_next)

            ###################
            ### directional ###
            ###################

            # intromonomer directional messages are regular intramonomer messages, fed through a dense net
            mA_ij_dir = self.directional_layers[i](mA_ij) # [e x 8]
            mB_ij_dir = self.directional_layers[i](mB_ij) # [e x 8]

            # contract with intramonomer unit vectors to make directional
            mA_ij_dir = tf.einsum('ex,em->exm', dRA_unit, mA_ij_dir) # [e x 3 x 8]
            mB_ij_dir = tf.einsum('ex,em->exm', dRB_unit, mB_ij_dir) # [e x 3 x 8]

            # sum directional messages to get directional atomic hidden states
            # NOTE: this summation must be linear to guarantee equivariance.
            #       because of this constraint, we applied a dense net before the summation, not after
            hA_dir = tf.math.unsorted_segment_sum(mA_ij_dir, e_AA_source, natomA) # [a x 3 x 8]
            hB_dir = tf.math.unsorted_segment_sum(mB_ij_dir, e_BB_source, natomB) # [a x 3 x 8]

            hA_dir_list.append(hA_dir)
            hB_dir_list.append(hB_dir)

        # concatenate hidden states over MP iterations
        hA = tf.keras.layers.Flatten()(tf.concat(hA_list, axis=-1))
        hB = tf.keras.layers.Flatten()(tf.concat(hB_list, axis=-1))

        # atom-pair features are a combo of atomic hidden states and the interatomic distance
        hAB = get_pair(hA, hB, rbf_sr, e_ABsr_source, e_ABsr_target)
        hBA = get_pair(hB, hA, rbf_sr, e_ABsr_target, e_ABsr_source)


        # project the directional atomic hidden states along the interatomic axis
        # (this is invariant to rotation)
        hA_dir = tf.concat(hA_dir_list, axis=-1)
        hB_dir = tf.concat(hB_dir_list, axis=-1)

        hA_dir_source = tf.gather(hA_dir, e_ABsr_source)
        hB_dir_target = tf.gather(hB_dir, e_ABsr_target)

        hA_dir_blah = tf.einsum('axf,ax->af', hA_dir_source, dR_sr_unit)
        hB_dir_blah = tf.einsum('axf,ax->af', hB_dir_target, -1.0 * dR_sr_unit)

        # concat projected directional hidden states to atom pair features
        hAB = tf.concat([hAB, hA_dir_blah, hB_dir_blah], axis=1)
        hBA = tf.concat([hBA, hB_dir_blah, hA_dir_blah], axis=1)

        # run atom-pair features through a dense net to predict SAPT components
        EAB_sr = tf.concat([self.readout_layer_elst(hAB), self.readout_layer_exch(hAB), self.readout_layer_ind(hAB), self.readout_layer_disp(hAB)], axis=1)
        EBA_sr = tf.concat([self.readout_layer_elst(hBA), self.readout_layer_exch(hBA), self.readout_layer_ind(hBA), self.readout_layer_disp(hBA)], axis=1)

        # symmetrize atom-pair predictions wrt monomers (E(a,b) = E(b,a)
        E_sr = EAB_sr + EBA_sr

        # scale atom-pair predictions by inverse distance
        cutoff = tf.math.reciprocal(dR_sr) ** 3
        E_sr = tf.einsum('xy,x->xy', E_sr, cutoff)

        # sum atom-pair predictions to get dimer predictions
        E_sr = tf.math.segment_sum(E_sr, dimer_ind)

        # padding necessary in case some dimers had zero short-range atom pairs
        dimer_padder_sr = tf.convert_to_tensor([[0,ndimer-tf.shape(E_sr)[0]], [0,0]])
        E_sr = tf.pad(E_sr, dimer_padder_sr)

        ####################################################
        ### predict multipole electrostatic interactions ###
        ####################################################

        # electrostatics between close atoms (we have to combine with NN IE)
        E_elst_sr = self.mtp_elst(qA, muA, quadA, qB, muB, quadB, e_ABsr_source, e_ABsr_target, dR_sr, dR_sr_xyz)
        E_elst_sr = tf.math.segment_sum(E_elst_sr, dimer_ind)
        E_elst_sr = tf.reshape(E_elst_sr, [-1, 1])
        E_elst_sr = tf.pad(E_elst_sr, dimer_padder_sr)

        # electrostatics between distance atoms (these atoms have zero NN IE)
        E_elst_lr = self.mtp_elst(qA, muA, quadA, qB, muB, quadB, e_ABlr_source, e_ABlr_target, dR_lr, dR_lr_xyz)
        E_elst_lr = tf.math.segment_sum(E_elst_lr, dimer_ind_lr)
        E_elst_lr = tf.reshape(E_elst_lr, [-1, 1])
        dimer_padder_lr = tf.convert_to_tensor([[0,ndimer-tf.shape(E_elst_lr)[0]], [0,0]])
        E_elst_lr = tf.pad(E_elst_lr, dimer_padder_lr)

        E_elst = tf.pad(E_elst_sr + E_elst_lr, tf.constant([[0,0], [0,3]]))

        #########################
        ### interation energy ###
        #########################

        return E_sr + E_elst


    def get_config(self):

        return {
            "atom_model" : self.atom_model,
            "n_message" : self.n_message,
            "n_rbf" : self.n_rbf,
            "n_neuron" : self.n_neuron,
            "n_embed" : self.n_embed,
            "r_cut_im" : self.r_cut_im,
        }

    @classmethod
    def from_config(cls, config):
        return cls(atom_model=config["atom_model"],
                   n_message=config["n_message"],
                   n_rbf=config["n_rbf"],
                   n_neuron=config["n_neuron"],
                   n_embed=config["n_embed"],
                   r_cut_im=config["r_cut_im"])

if __name__ == "__main__":

    model = KerasPairModel()
