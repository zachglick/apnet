import os
import scipy
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


tf.keras.backend.set_floatx('float64')

atom_num_dict = {'H' : 1,
             'C' : 6,
             'N' : 7,
             'O' : 8,
             'F' : 9,
             'P' : 15,
             'S' : 16,
             'Cl': 17,
             'Br': 35,
             }
elem_dict = {'H' : 0,
             'C' : 1,
             'N' : 2,
             'O' : 3,
             'F' : 4,
             'P' : 5,
             'S' : 6,
             'Cl': 7,
             'Br': 8,
             }

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

class GNN_layer(tf.keras.layers.Layer):
    def __init__(self, message_fn, update_fn, T):
        super(GNN_layer, self).__init__()
        self.message_fns = []
        for i in range(T):
            self.message_fns.append(message_fn([32,32], out_dim=32))
        self.update_fn = update_fn
        self.T = T

    @tf.function(experimental_relax_shapes=True)
    def call(self, h, e, x, q, mask):
        natom = e.shape[1]
        node_mask = tf.clip_by_value(tf.reduce_sum(mask, axis=1), clip_value_min=0, clip_value_max=1)
        for t in range(self.T):
            self.message_fn = self.message_fns[t]
            inp_atom_i = tf.concat([x, h, q], axis=-1)  # nmolec x natom x 9+32+1
            inp_i = tf.tile(tf.expand_dims(inp_atom_i, axis=2), [1, 1, natom, 1]) # nmolec x natom x natom x 9+32+1
            inp_j = tf.transpose(inp_i, [0, 2, 1, 3]) #nmolec x natom x natom x 9+32+1
            inp_ij = tf.concat([inp_i, inp_j, e], axis=-1) #nmolec x natom x natom x 9*2 + 32*2 + 1*2 + 32
            flat_inp_ij = tf.reshape(inp_ij, (-1, inp_ij.shape[-1]))#tf.shape(inp_ij)[-1])) #nmolec * natom**2 x 9*2+32*2+1*2+32

            flat_pair_messages = self.message_fn(flat_inp_ij)
            pair_messages = tf.reshape(flat_pair_messages, (-1, natom, natom, 32))
            messages = tf.reduce_sum(pair_messages, axis=2)
            update_input = tf.concat([h, messages], axis=2)
            masked_input = update_input * node_mask
            h = self.update_fn(masked_input)
            h = h * node_mask
        return h

#class EPN_layer(tf.keras.layers.Layer):
#    """Special 'Electron Passing Network,' which retains conservation of electrons but allows non-local passing"""
#
#    def __init__(self, pass_fn, T=1):
#        super(EPN_layer, self).__init__()
#        self.pass_fns = []
#        for t in range(T):
#            self.pass_fns.append(pass_fn([32,32]))
#        self.T = T
#
#    @tf.function(experimental_relax_shapes=True)
#    def call(self, h, e, x, q, mask):
#        
#        tol = tf.constant(1e-5)
#        clip = tf.clip_by_value(e, clip_value_min=tol, clip_value_max=1e5)
#        largest_e = tf.reduce_max(clip, axis=-1)
#        is_near = tf.math.not_equal(largest_e, tol)
#        is_near = tf.cast(is_near, dtype=tf.float32)
#
#        natom = e.shape[1]
#        mask = tf.cast(mask, dtype=tf.float32)
#        for t in range(self.T):
#            self.pass_fn = self.pass_fns[t]
#
#            inp_atom_i = tf.concat([x, h, q], axis=-1)  # nmolec x natom x 9+32+1
#            inp_i = tf.tile(tf.expand_dims(inp_atom_i, axis=2), [1, 1, natom, 1]) # nmolec x natom x natom x 9+32+1
#            inp_j = tf.transpose(inp_i, [0, 2, 1, 3]) #nmolec x natom x natom x 9+32+1
#            inp_ij_N = tf.concat([inp_i, inp_j, e], axis=-1) #nmolec x natom x natom x 9*2 + 32*2 + 1*2 + 32
#            inp_ij_T = tf.concat([inp_j, inp_i, e], axis=-1) #nmolec x natom x natom x 9*2 + 32*2 + 1*2 + 32
#
#            flat_inp_ij = tf.reshape(inp_ij_N, [-1, inp_ij_N.shape[-1]])
#            flat_inp_ji = tf.reshape(inp_ij_T, [-1, inp_ij_T.shape[-1]])
#
#            elec_ij_flat = self.pass_fn(flat_inp_ij)
#            elec_ji_flat = self.pass_fn(flat_inp_ji)
#
#            elec_ij = tf.reshape(elec_ij_flat, [-1, natom, natom])
#            elec_ji = tf.reshape(elec_ji_flat, [-1, natom, natom])
#
#            antisym_pass = 0.5 * (elec_ij - elec_ji) * tf.math.reduce_max(mask, axis=-1) * is_near
#
#            q += tf.expand_dims(tf.reduce_sum(antisym_pass, axis=2), axis=-1)
#        return q

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
        #print(clip.shape)
        largest_e = tf.reduce_max(clip, axis=-1)
        #print(largest_e.shape)
        is_near = tf.math.not_equal(largest_e, tol)
        is_near = tf.cast(is_near, dtype=tf.float64)
        #print(is_near.shape)

        #tf.print(e[0][0][-1])
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
            #tf.print(tf.shape(inp_i))
            
            inp_ij_N = tf.concat([inp_i, inp_j, e], axis=-1) #nmolec x natom x natom x 9*2 + 32*2 + 1*2 + 32
            inp_ij_T = tf.concat([inp_j, inp_i, e], axis=-1) #nmolec x natom x natom x 9*2 + 32*2 + 1*2 + 32

            flat_inp_ij = tf.reshape(inp_ij_N, [-1, inp_ij_N.shape[-1]])
            flat_inp_ji = tf.reshape(inp_ij_T, [-1, inp_ij_T.shape[-1]])
            #tf.print(tf.shape(flat_inp_ij))
            elec_ij_flat = self.pass_fn(flat_inp_ij)
            elec_ji_flat = self.pass_fn(flat_inp_ji)

            elec_ij = tf.reshape(elec_ij_flat, [-1, natom, natom])
            elec_ji = tf.reshape(elec_ji_flat, [-1, natom, natom])

            antisym_pass = 0.5 * (elec_ij - elec_ji) * exp_mask * is_near

            #tf.print(q[0])
            q += tf.expand_dims(tf.reduce_sum(antisym_pass, axis=2), axis=-1)

        #tf.print(mask[0])
        #tf.print(is_near[0])
        #tf.print(tf.reduce_sum(q, axis=[-1])[0])
        #tf.print(mask[0])
        #tf.print(tf.reduce_sum(q, axis=[-1])[-1])
        #tf.print(tf.reduce_sum(q, axis=[1,2]))
        return q


def get_init_edges(xyz, molecular_splits, num=32, cutoff=3.0, eta=2.0):
    mu = np.linspace(0.1, cutoff, num=num)
    D = scipy.spatial.distance_matrix(xyz,xyz)

    if molecular_splits.shape == (0,):
        adj = np.ones(D.shape)
    elif molecular_splits.shape == ():
        molec_vecA = np.zeros(D.shape[0])
        molec_vecA[:molecular_splits] = 1
        molec_vecB = np.zeros(D.shape[0])
        molec_vecB[molecular_splits:] = 1
        adj = np.outer(molec_vecA, molec_vecA.T) + np.outer(molec_vecB, molec_vecB.T)
    else:
        adj = np.zeros(D.shape)
        prev_split = 0
        for i, split in enumerate(molecular_splits):
            molec_vec = np.zeros(D.shape[0])
            molec_vec[prev_split:split] = 1
            print(molec_vec)
            molec_mat = np.outer(molec_vec, molec_vec.T)
            print(molec_mat)
            adj += molec_mat
        print(adj)
        exit()
    adj = np.expand_dims(adj, -1)

    C = (np.cos(np.pi * (D - 0.0) / cutoff) + 1.0) / 2.0

    C[D >= cutoff] = 0.0
    C[D <= 0.0] = 1.0
    np.fill_diagonal(C, 0.0)
    D = np.expand_dims(D, -1)
    D = np.tile(D, [1, 1, num])
    C = np.expand_dims(C, -1)
    C = np.tile(C, [1, 1, num])
    mu = np.expand_dims(mu, 0)
    mu = np.expand_dims(mu, 0)
    mu = np.tile(mu, [D.shape[0], D.shape[1], 1])
    e = C * np.exp(-eta * (D-mu)**2)
    e = np.array(e, dtype=np.float64)

    return e, C

def gen_init_state(path,  h_dim, e_dim):
    x = []
    h = []
    q = []
    Q = []
    e = []
    y = []
    for filename in os.listdir(path):
        if os.path.exists(path + filename[:-4] + ".npy") and filename.endswith(".xyz"):
            splits_path = path + filename[:-4] + "splits.npy"
            if os.path.exists(splits_path):
                splits = np.load(splits_path)
            else: splits = np.array([])
            xyzfile = open(path + filename, 'r')
            lines = xyzfile.readlines()
            y.append(np.array(np.load(path + filename[:-4] + '.npy'), dtype=np.float64))
            Q.append(np.array(lines[1].strip().split()[0], dtype=np.float64))
            xyz = []
            this_x = []
            for line in lines[2:]:
                data = line.split()
                elem_name = data[0]
                xyz.append([data[1], data[2], data[3]])
                ohe = np.zeros(len(elem_dict)+1)
                ohe[0] = atom_num_dict[elem_name]
                ohe[elem_dict[elem_name] + 1] = 1
                this_x.append(ohe)
            this_x = np.array(this_x)
            xyz = np.array(xyz, dtype=np.float64)
            x.append(np.array(this_x, dtype=np.float64))
            h.append(np.zeros((this_x.shape[0], h_dim), dtype=np.float64))
            avg_q = Q[-1] / len(this_x)
            q.append(np.array(np.ones(len(this_x)) * avg_q, dtype=np.float64))
            e.append(get_init_edges(xyz, splits, num=e_dim))
    return x, h, q, e, Q, y

def gen_flat_init_state(path,  h_dim, e_dim):
    x = []
    h = []
    q = []
    Q = []
    e = []
    y = []
    for filename in os.listdir(path):
        if os.path.exists(path + filename[:-4] + ".npy") and filename.endswith(".xyz"):
            xyzfile = open(path + filename, 'r')
            lines = xyzfile.readlines()
            y.append(np.array(np.load(path + filename[:-4] + '.npy'), dtype=np.float64))
            Q.append(np.array(lines[1].strip().split()[0], dtype=np.float64))
            xyz = []
            this_x = []
            for line in lines[2:]:
                data = line.split()
                elem_name = data[0]
                xyz.append([data[1], data[2], data[3]])
                ohe = np.zeros(len(elem_dict)+1)
                ohe[0] = atom_num_dict[elem_name]
                ohe[elem_dict[elem_name] + 1] = 1
                this_x.append(ohe)
            this_x = np.array(this_x)
            xyz = np.array(xyz, dtype=np.float64)
            these_edges = get_init_edges(xyz, num=e_dim)
            e.append(these_edges.reshape((-1, these_edges.shape[-1])))

            x.append(np.tile(np.array(this_x, dtype=np.float64), (these_edges.shape[0], 1)))
            h.append(np.tile(np.zeros((this_x.shape[0], h_dim), dtype=np.float64), (these_edges.shape[0], 1)))
            avg_q = Q[-1] / len(this_x)
            q.append(np.tile(np.array(np.ones((len(this_x), 1)) * avg_q, dtype=np.float64), (these_edges.shape[0], 1)))

    return x, h, q, e, Q, y

def gen_flat_padded_init_state(path,  h_dim, e_dim):
    x = []
    h = []
    q = []
    Q = []
    e = []
    y = []
    for filename in os.listdir(path):
        if os.path.exists(path + filename[:-4] + ".npy") and filename.endswith(".xyz"):
            xyzfile = open(path + filename, 'r')
            lines = xyzfile.readlines()
            y.append(np.array(np.load(path + filename[:-4] + '.npy'), dtype=np.float64))
            Q.append(np.array(lines[1].strip().split()[0], dtype=np.float64))
            xyz = []
            this_x = []
            for line in lines[2:]:
                data = line.split()
                elem_name = data[0]
                xyz.append([data[1], data[2], data[3]])
                ohe = np.zeros(len(elem_dict)+1)
                ohe[0] = atom_num_dict[elem_name]
                ohe[elem_dict[elem_name] + 1] = 1
                this_x.append(ohe)
            this_x = np.array(this_x)
            xyz = np.array(xyz, dtype=np.float64)
            these_edges = get_init_edges(xyz, num=e_dim)
            e.append(these_edges.reshape((-1, these_edges.shape[-1])))

            x.append(np.tile(np.array(this_x, dtype=np.float64), (these_edges.shape[0], 1)))
            h.append(np.tile(np.zeros((this_x.shape[0], h_dim), dtype=np.float64), (these_edges.shape[0], 1)))
            avg_q = Q[-1] / len(this_x)
            q.append(np.tile(np.array(np.ones((len(this_x), 1)) * avg_q, dtype=np.float64), (these_edges.shape[0], 1)))
    
    largest_system = np.max([y[i].shape[0] for i in range(len(y))])
    x_padded = np.zeros((len(Q), largest_system**2, x[0].shape[1]))
    h_padded = np.zeros((len(Q), largest_system**2, h[0].shape[1]))
    q_padded = np.zeros((len(Q), largest_system**2, q[0].shape[1]))
    e_padded = np.zeros((len(Q), largest_system**2, e[0].shape[1]))
    y_padded = np.zeros((len(Q), largest_system, y[0].shape[1]))
    pad = np.zeros((len(Q)))

    for i in range(x_padded.shape[0]):
        for j in range(y[i].shape[0]):
            y_padded[i][j] = y[i][j]
            pad[i] = j
        for j in range(x[i].shape[0]):
            for k in range(x[i][j].shape[0]):
                x_padded[i][j][k] = x[i][j][k]
            for k in range(h[i][j].shape[0]):
                h_padded[i][j][k] = h[i][j][k]
            for k in range(q[i][j].shape[0]):
                q_padded[i][j][k] = q[i][j][k]
            for k in range(e[i][j].shape[0]):
                e_padded[i][j][k] = e[i][j][k]
    return x_padded, h_padded, q_padded, e_padded, Q, y_padded, pad

def gen_padded_init_state(path,  h_dim, e_dim):
    x = []
    h = []
    q = []
    Q = []
    e = []
    y = []
    soft_mask = []
    names = []
    for filename in os.listdir(path):
        label_file = path + filename[:-4] + '.npy'
        #if os.path.exists(label_file) and filename.endswith(".xyz"):
        if filename.endswith(".xyz"):
            splits_path = path + filename[:-4] + "splits.npy"
            if os.path.exists(splits_path):
                splits = np.load(splits_path)
            else: splits = np.array([])
            xyzfile = open(path + filename, 'r')
            lines = xyzfile.readlines()
            label_file = path + filename[:-4] + '.npy'
            if os.path.exists(label_file):
                y.append(np.array(np.load(label_file), dtype=np.float64))
            else:
                print('No labels provided, y set to 0')
                y.append(np.zeros(len(lines)-2))
            Q.append(np.array(lines[1].strip().split()[0], dtype=np.float64))
            names.append(filename[:-4])
            xyz = []
            this_x = []
            for line in lines[2:]:
                data = line.split()
                elem_name = data[0]
                xyz.append([data[1], data[2], data[3]])
                ohe = np.zeros(len(elem_dict)+1)
                ohe[0] = atom_num_dict[elem_name]
                ohe[elem_dict[elem_name] + 1] = 1
                this_x.append(ohe)
            this_x = np.array(this_x)
            xyz = np.array(xyz, dtype=np.float64)
            these_edges, this_soft_mask = get_init_edges(xyz, splits, num=e_dim)
            e.append(these_edges.reshape((-1, these_edges.shape[-1])))
            soft_mask.append(this_soft_mask)

            x.append(np.tile(np.array(this_x, dtype=np.float64), (these_edges.shape[0], 1)))
            h.append(np.tile(np.zeros((this_x.shape[0], h_dim), dtype=np.float64), (these_edges.shape[0], 1)))
            avg_q = Q[-1] / len(this_x)
            q.append(np.tile(np.array(np.ones((len(this_x), 1)) * avg_q, dtype=np.float64), (these_edges.shape[0], 1)))

    largest_system = np.max([y[i].shape[0] for i in range(len(y))])

    x_padded = np.zeros((len(Q), largest_system, largest_system,  x[0].shape[1]))
    h_padded = np.zeros((len(Q), largest_system, largest_system, h[0].shape[1]))
    q_padded = np.zeros((len(Q), largest_system, largest_system, q[0].shape[1]))
    e_padded = np.zeros((len(Q), largest_system, largest_system, e[0].shape[1]))
    soft_mask_padded = np.zeros((len(Q), largest_system, largest_system, 1))
    y_padded = np.zeros((len(Q), largest_system, 1))
    pad_n = np.zeros((len(Q)))
    mask = np.zeros((len(Q), largest_system, largest_system))


    for i in range(x_padded.shape[0]):
        molec_size = np.sqrt(x[i].shape[0]).astype(np.int32)
        for j in range(y[i].shape[0]):
            y_padded[i][j] = y[i][j]
            #soft_mask_padded[i][j] = soft_mask[i][j]
            pad_n[i] = j
        for j in range(molec_size):
            for k in range(molec_size):
                x_padded[i][j][k] = x[i][j*molec_size + k]
                h_padded[i][j][k] = h[i][j*molec_size + k]
                q_padded[i][j][k] = q[i][j*molec_size + k]
                e_padded[i][j][k] = e[i][j*molec_size + k]
                mask[i][j][k] = 1

    return x_padded, h_padded, q_padded, e_padded, Q, y_padded, mask, np.array(names)


def make_model(layers, h_dim, T, n_elems, natom):#mask, natom):                    # mask: nmol x natom
    message_model = MLP_layer
    update_fn = MLP_layer(layers, out_dim=h_dim)
    graph_net = GNN_layer(message_model, update_fn, T)
    electron_model = MLP_layer
    electron_net = EPN_layer(electron_model, T=T)

    h_inp = tf.keras.Input(shape=(natom, natom, h_dim), dtype='float64', name='h_inp')          # nmol x natom x natom x h_dim
    e_inp = tf.keras.Input(shape=(natom, natom, h_dim), dtype='float64', name='e_inp')          # nmol x natom x natom x h_dim
    x_inp = tf.keras.Input(shape=(natom, natom, n_elems), dtype='float64', name='x_inp')        # nmol x natom x natom x n_elems
    q_inp = tf.keras.Input(shape=(natom, natom, 1), dtype='float64', name='q_inp')              # nmol x natom x natom x 1
    mask_inp = tf.keras.Input(shape=(natom, natom, 1), dtype='float64', name='mask_inp')        # nmol x natom x natom x 1
    
    h = tf.math.divide_no_nan(tf.math.reduce_sum(h_inp, axis=1), tf.math.reduce_sum(mask_inp, axis=1))
    x = tf.math.divide_no_nan(tf.math.reduce_sum(x_inp, axis=1), tf.math.reduce_sum(mask_inp, axis=1))
    q = tf.math.divide_no_nan(tf.math.reduce_sum(q_inp, axis=1), tf.math.reduce_sum(mask_inp, axis=1))
    
    graph_feats = graph_net(h, e_inp, x, q, mask_inp)                                     # nmol x natom x h_dim
    q_pred = electron_net(graph_feats, e_inp, x, q, mask_inp)                             # nmol x natom x 1

    model = tf.keras.Model(inputs=[h_inp, e_inp, x_inp, q_inp, mask_inp], outputs=q_pred)

    return model

@tf.function(experimental_relax_shapes=True)
def train_step(h, e, x, q, y, mask):
    with tf.GradientTape() as tape:
        predictions = model([h, e, x, q, mask])
        loss = tf.keras.losses.MSE(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_acc(predictions, y)
    return predictions

@tf.function(experimental_relax_shapes=True)
def test_step(h, e, x, q, y, mask):
    predictions = model([h, e, x, q, mask])
    t_loss = tf.keras.losses.MSE(y, predictions)
    test_loss(t_loss)
    test_acc(predictions, y)
    return predictions

if __name__ == "__main__":
    h_dim = 48
    e_dim = 48
    layers = [32, 32]
    T = 5
    path = 'data/mixed/'
    n_elems = 10
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.MeanAbsoluteError(name='train_acc')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc = tf.keras.metrics.MeanAbsoluteError(name='test_acc')
    EPOCHS = 500
    best_test_acc = np.inf

    x, h, q, e, Q, y, mask, names = gen_padded_init_state(path, h_dim, e_dim)

    model = make_model(layers, h_dim, T, n_elems, x.shape[1])

    xt, xe, ht, he, qt, qe, et, ee, Qt, Qe, yt, ye, maskt, maske, namest, namese = train_test_split(x,h,q,e,Q,y,mask,names, test_size=0.2, random_state=42)
    
    np.save("train_names.npy", namest, allow_pickle=True)
    np.save("val_names.npy", namese, allow_pickle=True)

    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_acc.reset_states()
        test_loss.reset_states()
        test_acc.reset_states()
        train_preds = []
        test_preds = []
        for i in range(len(xt)):
            hb = np.array(np.expand_dims(ht[i], axis=0))
            eb = np.array(np.expand_dims(et[i], axis=0))
            xb = np.array(np.expand_dims(xt[i], axis=0))
            qb = np.array(np.expand_dims(qt[i], axis=0))
            yb = np.array(np.expand_dims(yt[i], axis=0))
            maskb = np.array(np.expand_dims(maskt[i], axis=0))
            #train_preds.append(train_step(ht[i], et[i], xt[i], qt[i], yt[i], maskt[i]))
            train_preds.append(train_step(hb, eb, xb, qb, yb, maskb))
        for i in range(len(xe)):
            hb = np.array(np.expand_dims(he[i], axis=0))
            eb = np.array(np.expand_dims(ee[i], axis=0))
            xb = np.array(np.expand_dims(xe[i], axis=0))
            qb = np.array(np.expand_dims(qe[i], axis=0))
            yb = np.array(np.expand_dims(ye[i], axis=0))
            maskb = np.array(np.expand_dims(maske[i], axis=0))
            test_preds.append(test_step(hb, eb, xb, qb, yb, maskb))
        if test_acc.result() < best_test_acc:
            best_test_acc = test_acc.result()
            model.save_weights('models/model_weights')
        #    model.save(f'models/model')
        #    #tf.saved_model.save(model(he[-1], ee[-1], xe[-1], qe[-1]), 'models/model')
            np.save("train_pred_charges.npy", np.array(np.squeeze(train_preds)))
            np.save("train_lab_charges.npy", np.squeeze(yt))
            np.save("test_pred_charges.npy", np.array(np.squeeze(test_preds)))
            np.save("test_lab_charges.npy", np.squeeze(ye))

        template = 'Epoch {}, Loss: {}, Acc: {}, Test Loss: {}, Test Acc: {}'
        print(template.format(epoch, train_loss.result(), train_acc.result(), test_loss.result(), test_acc.result()))
