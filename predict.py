import os, sys, argparse
import numpy as np
import pandas as pd

print("Loading TF...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

import util

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test one or many atomic multipole models')
    parser.add_argument('data',
                        help='Dataset for testing (must be in ./data/)')

    args = parser.parse_args(sys.argv[1:])

    data = args.data

    atom_modelnames = ['hfadz1', 'hfadz2', 'hfadz3']
    for atom_modelname in atom_modelnames:
        if not os.path.isfile(f'atom_models/{atom_modelname}.hdf5'):
            print(f'Model atom_models/{atom_modelname}.hdf5 does not exist!')
            exit()

    # these are defined by the dataset
    pad_dim = 55
    nelem = 36
    
    # this is up to the user
    nembed = 10
    nnodes = [256,128,64]
    nmessage = 3

    # make the model 
    mus = np.linspace(0.8, 5.0, 43)
    etas = np.array([-100.0] * 43)
    RA, RB, ZA, ZB, aQA, aQB = util.get_dimer_data(f'./data/{data}.pkl', pad_dim)

    print('Shapes')
    print(RA.shape, ZA.shape, aQA.shape)
    print(RB.shape, ZB.shape, aQB.shape)
    
    atom_models = []
    for atom_modelname in atom_modelnames:
        atom_models.append(util.get_model(mus, etas, pad_dim, nelem, nembed, nnodes, nmessage))
        atom_models[-1].load_weights(f'./atom_models/{atom_modelname}.hdf5')
    
    qA_preds = []
    muA_preds = []
    Q_iiA_preds = []
    Q_ijA_preds = []
    for atom_model in atom_models:
        yA_pred = atom_model.predict([RA, ZA, aQA])
        qA_preds.append(yA_pred[0])
        muA_preds.append(yA_pred[1])
        Q_iiA_preds.append(yA_pred[2])
        Q_ijA_preds.append(yA_pred[3])
    
    qA_avg = np.expand_dims(np.average(np.array(qA_preds), axis=0), axis=-1)
    muA_avg = np.average(np.array(muA_preds), axis=0)
    Q_iiA_avg = np.average(np.array(Q_iiA_preds), axis=0)
    Q_ijA_avg = np.average(np.array(Q_ijA_preds), axis=0)

    avgA = np.concatenate([qA_avg, muA_avg, Q_iiA_avg, Q_ijA_avg], axis=-1)
    listA1 = list(avgA)

    df = pd.read_pickle(f"./data/{data}.pkl")
    listA2 = df.multipoles_A.tolist()
    assert len(listA1) == len(listA2)
    for ai in range(len(listA1)):
        assert listA1[ai].shape == listA2[ai].shape
        print(np.max(np.abs(listA1[ai] - listA2[ai])))
    del df

    qB_preds = []
    muB_preds = []
    Q_iiB_preds = []
    Q_ijB_preds = []
    for atom_model in atom_models:
        yB_pred = atom_model.predict([RB, ZB, aQB])
        qB_preds.append(yB_pred[0])
        muB_preds.append(yB_pred[1])
        Q_iiB_preds.append(yB_pred[2])
        Q_ijB_preds.append(yB_pred[3])
    
    qB_avg = np.expand_dims(np.average(np.array(qB_preds), axis=0), axis=-1)
    muB_avg = np.average(np.array(muB_preds), axis=0)
    Q_iiB_avg = np.average(np.array(Q_iiB_preds), axis=0)
    Q_ijB_avg = np.average(np.array(Q_ijB_preds), axis=0)
    
    avgB = np.concatenate([qB_avg, muB_avg, Q_iiB_avg, Q_ijB_avg], axis=-1)
    listB1 = list(avgB)

    df = pd.read_pickle(f"./data/{data}.pkl")
    listB2 = df.multipoles_B.tolist()

    assert len(listB1) == len(listB2)
    for bi in range(len(listB1)):
        assert listB1[bi].shape == listB2[bi].shape
        print(np.max(np.abs(listB1[bi] - listB2[bi])))
    del df
