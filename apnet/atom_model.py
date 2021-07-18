import sys, os
import math
import time
import numpy as np

from pathlib import Path
ROOT_DIR = Path(__file__).parent

import tensorflow as tf

from apnet.keras_atom_model import KerasAtomModel
from apnet import constants

class AtomDataLoader:
    """ todo """

    def __init__(self, molecules, multipoles, r_cut):

        self.r_cut = r_cut

        self.R_list = []
        self.Z_list = []
        self.total_charge_list = []
        self.e_source_list = []
        self.e_target_list = []

        for molecule in molecules:

            R, Z, total_charge = self.molecule_to_data(molecule)
            e_source, e_target = self.edges(R)

            self.R_list.append(R)
            self.Z_list.append(Z)
            self.total_charge_list.append(total_charge)
            self.e_source_list.append(e_source)
            self.e_target_list.append(e_target)

        self.charge_list = []
        self.dipole_list = []
        self.qpole_list = []

        if multipoles is not None:

            self.has_multipoles = True

            for multipole in multipoles:
                self.charge_list.append(multipole[:,0])
                self.dipole_list.append(multipole[:,1:4])
                self.qpole_list.append(self.make_quad(multipole[:,4:10]))

        else:

            self.has_multipoles = False

    def get_data(self, inds):

        inp = { "R" : [],
                "Z" : [],
                "e_source" : [],
                "e_target" : [],
                "total_charge" : [],
                "molecule_ind" : [],
              }

        offset = 0
        for i, ind in enumerate(inds):
            inp["R"].append(self.R_list[ind])
            inp["Z"].append(self.Z_list[ind])
            inp["e_source"].append(self.e_source_list[ind] + offset)
            inp["e_target"].append(self.e_target_list[ind] + offset)
            inp["total_charge"].append([self.total_charge_list[ind]])
            inp["molecule_ind"].append(np.full(len(self.R_list[ind]), i))
            offset += self.R_list[ind].shape[0]

        for k, v in inp.items():
            inp[k] = np.concatenate(v, axis=0)

        if not self.has_multipoles:
            return inp

        target_charge = np.concatenate([self.charge_list[ind] for ind in inds], axis=0)
        target_dipole = np.concatenate([self.dipole_list[ind] for ind in inds], axis=0)
        target_qpole = np.concatenate([self.qpole_list[ind] for ind in inds], axis=0)

        return inp, target_charge, target_dipole, target_qpole

    def edges(self, R):
    
        natom = np.shape(R)[0]
    
        RA = np.expand_dims(R, 0)
        RB = np.expand_dims(R, 1)
    
        RA = np.tile(RA, [natom,1,1])
        RB = np.tile(RB, [1,natom,1])
    
        dist = np.linalg.norm(RA - RB, axis=2)
    
        mask = np.logical_and(dist < self.r_cut, dist > 0.0)
        edges = np.where(mask) # dimensions [n_edge x 2]
    
        return edges[0], edges[1]

    def make_quad(self, flat_quad):
    
        natom = flat_quad.shape[0]
        full_quad = np.zeros((natom, 3, 3))
        full_quad[:,0,0] = flat_quad[:,0] # xx
        full_quad[:,0,1] = flat_quad[:,1] # xy
        full_quad[:,1,0] = flat_quad[:,1] # xy
        full_quad[:,0,2] = flat_quad[:,2] # xz
        full_quad[:,2,0] = flat_quad[:,2] # xz
        full_quad[:,1,1] = flat_quad[:,3] # yy
        full_quad[:,1,2] = flat_quad[:,4] # yz
        full_quad[:,2,1] = flat_quad[:,4] # yz
        full_quad[:,2,2] = flat_quad[:,5] # zz
    
        trace = full_quad[:,0,0] + full_quad[:,1,1] + full_quad[:,2,2]
    
        full_quad[:,0,0] -= trace / 3.0
        full_quad[:,1,1] -= trace / 3.0
        full_quad[:,2,2] -= trace / 3.0
    
        return full_quad


    def molecule_to_data(self, molecule):
        """ QCelemental molecule to ML-ready numpy arrays """

        # this better be a molecule 
        if  len(molecule.fragments) != 1:
            raise AssertionError(f"A molecule must have exactly 1 molecular fragment, found {len(molecule.fragments)}")

        R = np.array(molecule.geometry, dtype=np.float32) * constants.au2ang
        # todo: int
        Z = np.array([constants.elem_to_z[z] for z in molecule.symbols], dtype=np.float32)
        total_charge = int(molecule.molecular_charge)

        return (R, Z, total_charge)


class AtomModel:
    """ todo """

    def __init__(self, **kwargs):

        # todo : pass params
        self.model = KerasAtomModel()

    @classmethod
    def from_file(cls, model_path):

        obj = cls()
        obj.model = tf.keras.models.load_model(model_path)
        return obj

    @classmethod
    def pretrained(cls, index=0):

        obj = cls()
        model_path = f"{ROOT_DIR}/atom_models/atom{index}"
        obj.model = tf.keras.models.load_model(model_path)
        return obj

    def train(self, molecules_t, multipoles_t, molecules_v, multipoles_v, model_path=None, log_path=None, **kwargs):

        # redirect stdout to log file, if specified
        if log_path is not None:
            default_stdout = sys.stdout
            log_file = open(log_path, "a")
            sys.stdout = log_file

        # refuse to overwrite an existing model file, if specified
        if model_path is not None:
            if os.path.exists(model_path):
                raise Exception(f"{model_path=} already exists. Delete existing model or choose a new `model_path`")

        print("~~ Training Atom Model ~~", flush=True)
        # todo : print time and date. maybe machine specs?

        if model_path is not None:
            print(f"\nSaving model to '{model_path}'", flush=True)
        else:
            print("\nNo `model_path` provided, not saving model")

        # network hyperparameters
        n_message = kwargs.get("n_message", 3)
        n_neuron = kwargs.get("n_neuron", 128)
        n_embed = kwargs.get("n_embed", 8)
        n_rbf = kwargs.get("n_rbf", 8)
        r_cut = kwargs.get("r_cut", 5.0)

        print("\nNetwork Hyperparameters:", flush=True)
        print(f"  {n_message=}", flush=True)
        print(f"  {n_neuron=}", flush=True)
        print(f"  {n_embed=}", flush=True)
        print(f"  {n_rbf=}", flush=True)
        print(f"  {r_cut=}", flush=True)
        
        # training hyperparameters
        n_epochs = kwargs.get("n_epochs", 15)
        batch_size = kwargs.get("batch_size", 16)
        learning_rate = kwargs.get("learning_rate", 0.0005)
        learning_rate_decay = 0.0 #TODO

        print("\nTraining Hyperparameters:", flush=True)
        print(f"  {n_epochs=}", flush=True)
        print(f"  {batch_size=}", flush=True)
        print(f"  {learning_rate=}", flush=True)
        print(f"  {learning_rate_decay=}", flush=True)

        Nt = len(molecules_t)
        Nv = len(molecules_v)

        print("\nDataset:", flush=True)
        print(f"  n_molecules_train={Nt}", flush=True)
        print(f"  n_molecules_val={Nv}", flush=True)

        inds_t = np.arange(Nt)
        inds_v = np.arange(Nv)

        np.random.seed(4201)
        np.random.shuffle(inds_t)
        num_batches = math.ceil(Nt / batch_size)

        # TODO: replaced hardcoded 200 molecules. Probably want a data_loader.get_large_batch
        inds_t_chunks = [inds_t[i*200:min((i+1)*200,Nt)] for i in range(math.ceil(Nt / 200))]
        inds_v_chunks = [inds_v[i*200:min((i+1)*200,Nv)] for i in range(math.ceil(Nv / 200))]

        print("\nProcessing Dataset...", flush=True)
        time_loaddata_start = time.time()
        data_loader_t = AtomDataLoader(molecules_t, multipoles_t, r_cut)
        data_loader_v = AtomDataLoader(molecules_v, multipoles_v, r_cut)
        dt_loaddata = time.time() - time_loaddata_start
        print(f"...Done in {dt_loaddata:.2f} seconds", flush=True)

        inp_t, charge_t, dipole_t, qpole_t = data_loader_t.get_data(inds_t)
        inp_v, charge_v, dipole_v, qpole_v = data_loader_v.get_data(inds_v)

        inp_t_chunks = [data_loader_t.get_data(inds_t_i) for inds_t_i in inds_t_chunks]
        inp_v_chunks = [data_loader_v.get_data(inds_v_i) for inds_v_i in inds_v_chunks]

        preds_all_t = [test_batch(self.model, inp_t_i[0]) for inp_t_i in inp_t_chunks]
        preds_charge_t = np.concatenate([pred[0] for pred in preds_all_t], axis=0)
        preds_dipole_t = np.concatenate([pred[1] for pred in preds_all_t], axis=0)
        preds_qpole_t = np.concatenate([pred[2] for pred in preds_all_t], axis=0)

        preds_all_v = [test_batch(self.model, inp_v_i[0]) for inp_v_i in inp_v_chunks]
        preds_charge_v = np.concatenate([pred[0] for pred in preds_all_v], axis=0)
        preds_dipole_v = np.concatenate([pred[1] for pred in preds_all_v], axis=0)
        preds_qpole_v = np.concatenate([pred[2] for pred in preds_all_v], axis=0)

        mae_charge_t = np.average(np.abs(np.array(preds_charge_t) - charge_t))
        mae_charge_v = np.average(np.abs(np.array(preds_charge_v) - charge_v))

        mae_dipole_t = np.average(np.abs(np.array(preds_dipole_t) - dipole_t))
        mae_dipole_v = np.average(np.abs(np.array(preds_dipole_v) - dipole_v))

        mae_qpole_t = np.average(np.abs(np.array(preds_qpole_t) - qpole_t))
        mae_qpole_v = np.average(np.abs(np.array(preds_qpole_v) - qpole_v))

        mse_charge_t = np.average(np.square(np.array(preds_charge_t) - charge_t))
        mse_charge_v = np.average(np.square(np.array(preds_charge_v) - charge_v))

        mse_dipole_t = np.average(np.square(np.array(preds_dipole_t) - dipole_t))
        mse_dipole_v = np.average(np.square(np.array(preds_dipole_v) - dipole_v))

        mse_qpole_t = np.average(np.square(np.array(preds_qpole_t) - qpole_t))
        mse_qpole_v = np.average(np.square(np.array(preds_qpole_v) - qpole_v))

        loss_v_best = mse_charge_v + mse_dipole_v + mse_qpole_v

        print("                                         Charge          Dipole        Quadrupole", flush=True)
        print(f"  (Pre-training)                MAE: {mae_charge_t:>7.3f}/{mae_charge_v:<7.3f} {mae_dipole_t:>7.3f}/{mae_dipole_v:<7.3f} {mae_qpole_t:>7.3f}/{mae_qpole_v:<7.3f}", flush=True)

        if model_path is not None:
            self.model.save(model_path)

        # todo : learning rate decay
        if False:
            learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=(num_batches * 60), decay_rate=0.5, staircase=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_scheduler)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        loss_fn = tf.keras.losses.MSE

        for ep in range(n_epochs):

            t1 = time.time()

            preds_charge_t, preds_dipole_t, preds_qpole_t = [], [], []
            err_t, err_dipole_t, err_qpole_t = [], [], []

            for batch in range(num_batches):
                batch_start = batch_size * batch
                inds_batch = inds_t[batch_start:min(Nt,batch_start+batch_size)]

                inp_batch, charge_batch, dipole_batch, qpole_batch = data_loader_t.get_data(inds_batch)

                preds_batch, preds_dipole_batch, preds_qpole_batch = train_batch(self.model, optimizer, loss_fn, inp_batch, charge_batch, dipole_batch, qpole_batch)
                preds_batch = tf.reshape(preds_batch, -1)

                preds_charge_t.append(preds_batch)
                err_t.append(preds_batch - charge_batch)

                preds_dipole_t.append(preds_dipole_batch)
                err_dipole_t.append(preds_dipole_batch - dipole_batch)

                preds_qpole_t.append(preds_qpole_batch)
                err_qpole_t.append(preds_qpole_batch - qpole_batch)

            preds_charge_t = np.concatenate(preds_charge_t)
            err_t = np.concatenate(err_t)
            mae_charge_t = np.average(np.abs(err_t))

            preds_dipole_t = np.concatenate(preds_dipole_t)
            err_dipole_t = np.concatenate(err_dipole_t)
            mae_dipole_t = np.average(np.abs(err_dipole_t))

            preds_qpole_t = np.concatenate(preds_qpole_t)
            err_qpole_t = np.concatenate(err_qpole_t)
            mae_qpole_t = np.average(np.abs(err_qpole_t))

            preds_all_v = [test_batch(self.model, inp_v_i[0]) for inp_v_i in inp_v_chunks]
            preds_charge_v = np.concatenate([pred[0] for pred in preds_all_v], axis=0)
            preds_dipole_v = np.concatenate([pred[1] for pred in preds_all_v], axis=0)
            preds_qpole_v = np.concatenate([pred[2] for pred in preds_all_v], axis=0)

            mae_charge_v = np.average(np.abs(np.array(preds_charge_v) - charge_v))
            mae_dipole_v = np.average(np.abs(np.array(preds_dipole_v) - dipole_v))
            mae_qpole_v = np.average(np.abs(np.array(preds_qpole_v) - qpole_v))

            mse_charge_v = np.average(np.square(np.array(preds_charge_v) - charge_v))
            mse_dipole_v = np.average(np.square(np.array(preds_dipole_v) - dipole_v))
            mse_qpole_v = np.average(np.square(np.array(preds_qpole_v) - qpole_v))

            loss_v = mse_charge_v + mse_dipole_v + mse_qpole_v

            np.random.shuffle(inds_t)

            dt = time.time() - t1

            if loss_v < loss_v_best:
                if model_path is not None:
                    self.model.save(model_path)
                loss_v_best = loss_v
                improved = "*"
            else:
                improved = ""

            print(f"  EPOCH: {ep:4d} ({dt:<7.2f} sec)     MAE: {mae_charge_t:>7.3f}/{mae_charge_v:<7.3f} {mae_dipole_t:>7.3f}/{mae_dipole_v:<7.3f} {mae_qpole_t:>7.3f}/{mae_qpole_v:<7.3f} {improved}", flush=True)
        
        preds_all_t = [test_batch(self.model, inp_t_i[0]) for inp_t_i in inp_t_chunks]
        preds_charge_t = np.concatenate([pred[0] for pred in preds_all_t], axis=0)
        preds_dipole_t = np.concatenate([pred[1] for pred in preds_all_t], axis=0)
        preds_qpole_t = np.concatenate([pred[2] for pred in preds_all_t], axis=0)

        preds_all_v = [test_batch(self.model, inp_v_i[0]) for inp_v_i in inp_v_chunks]
        preds_charge_v = np.concatenate([pred[0] for pred in preds_all_v], axis=0)
        preds_dipole_v = np.concatenate([pred[1] for pred in preds_all_v], axis=0)
        preds_qpole_v = np.concatenate([pred[2] for pred in preds_all_v], axis=0)

        mae_charge_t = np.average(np.abs(np.array(preds_charge_t) - charge_t))
        mae_charge_v = np.average(np.abs(np.array(preds_charge_v) - charge_v))

        mae_dipole_t = np.average(np.abs(np.array(preds_dipole_t) - dipole_t))
        mae_dipole_v = np.average(np.abs(np.array(preds_dipole_v) - dipole_v))

        mae_qpole_t = np.average(np.abs(np.array(preds_qpole_t) - qpole_t))
        mae_qpole_v = np.average(np.abs(np.array(preds_qpole_v) - qpole_v))


        if log_path is not None:
            sys.stdout = default_stdout
            log_file.close()

    def predict(self, molecules):

        N = len(molecules)

        inds = np.arange(N)
        # TODO: replaced hardcoded 200 molecules. Probably want a data_loader.get_large_batch

        inds_chunks = [inds[i*200:min((i+1)*200,N)] for i in range(math.ceil(N / 200))]

        print("Processing Dataset...", flush=True)
        time_loaddata_start = time.time()
        data_loader = AtomDataLoader(molecules, None, self.model.get_config()["r_cut"])
        dt_loaddata = time.time() - time_loaddata_start
        print(f"...Done in {dt_loaddata:.2f} seconds", flush=True)

        print("\nPredicting Atomic Properties...", flush=True)
        time_predprop_start = time.time()
        inp_chunks = [data_loader.get_data(inds_i) for inds_i in inds_chunks]
        preds_all = [test_batch(self.model, inp_i) for inp_i in inp_chunks]

        preds_charge = np.concatenate([chunk[0] for chunk in preds_all], axis=0)
        preds_dipole = np.concatenate([chunk[1] for chunk in preds_all], axis=0)
        preds_qpole = np.concatenate([chunk[2] for chunk in preds_all], axis=0)

        atom_ind_ranges = np.cumsum(np.concatenate([[0], np.array([Z.shape[0] for Z in data_loader.Z_list])]))

        multipoles = []

        for i in range(N):
            atom_start = atom_ind_ranges[i]
            atom_end = atom_ind_ranges[i+1]

            mol_charge = preds_charge[atom_start:atom_end].reshape(-1,1)
            mol_dipole = preds_dipole[atom_start:atom_end]
            mol_qpole = preds_qpole[atom_start:atom_end].reshape(-1,9)[:,[0,1,2,4,5,8]]

            mol_multipole = np.concatenate([mol_charge, mol_dipole, mol_qpole], axis=1)

            multipoles.append(mol_multipole)

        dt_predprop = time.time() - time_predprop_start
        print(f"Done in {dt_predprop:.2f} seconds", flush=True)

        return np.array(multipoles)


    # Possible TODO: predict_elst, transfer_learning, gradient


@tf.function(experimental_relax_shapes=True)
def train_batch(model, optimizer, loss_fn, inp_t, target_t, target_dipole_t, target_qpole_t):

    target_t = tf.cast(target_t, tf.float32)

    with tf.GradientTape() as tape:

        preds_charge_t, preds_dipole_t, preds_qpole_t, hidden_list_t = model(inp_t, training=True)
        preds_charge_t = tf.reshape(preds_charge_t, [-1])
        loss_value = loss_fn(target_t, preds_charge_t)
        loss_value += tf.math.reduce_mean(loss_fn(target_dipole_t, preds_dipole_t))
        loss_value += tf.math.reduce_mean(loss_fn(target_qpole_t, preds_qpole_t))

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return preds_charge_t, preds_dipole_t, preds_qpole_t

@tf.function(experimental_relax_shapes=True)
def test_batch(model, inp_v):

    preds_charge_v, preds_dipole_v, preds_qpole_v, hidden_list_v = model(inp_v, training=False)
    preds_charge_v = tf.reshape(preds_charge_v, [-1])

    return preds_charge_v, preds_dipole_v, preds_qpole_v

def total_charge_error(total_charge, pred_atom_charge, molecule_ind):

    pred_total_charge = np.zeros_like(total_charge, dtype=np.float64)
    for atom_ind, pred_charge in enumerate(pred_atom_charge):
        pred_total_charge[molecule_ind[atom_ind]] += pred_charge
    #print(pred_total_charge)
    return np.average(np.abs(pred_total_charge - total_charge))


if __name__ == "__main__":
    
    model = KerasAtomModel()
    model2 = AtomModel()
