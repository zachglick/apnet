import os, sys, argparse
import numpy as np
import pandas as pd

print("Loading TF...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Add, Concatenate, Dense, Embedding, Flatten, Input, InputLayer, Lambda, Layer, Reshape, Subtract
tf.keras.backend.set_floatx('float64')

import util

elem_z = {
        'H'  : 1,
        'C'  : 6,
        'N'  : 7,
        'O'  : 8,
        'F'  : 9,
        'NA' : 11,
        'P'  : 15,
        'S'  : 16,
        'CL' : 17,
        'BR' : 35,
        }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test one or many atomic multipole models')
    parser.add_argument('data',
                        help='Dataset for testing (must be in ./data/)')
    parser.add_argument('modelnames', nargs='+',
                        help='Name for saving model')
    args = parser.parse_args(sys.argv[1:])

    for modelname in args.modelnames:
        if not os.path.isfile(f'atom_models/{modelname}.hdf5'):
            print(f'Model atom_models/{modelname}.hdf5 does not exist!')
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
    
    models = []
    model_names = ['hfadz1', 'hfadz2', 'hfadz3']
    print("Loading Models...")
    for model_name in model_names:
        models.append(util.get_model(mus, etas, pad_dim, nelem, nembed, nnodes, nmessage))
        models[-1].load_weights(f'./atom_models/{model_name}.hdf5')

    print("Loading Dataset...")
    RA, RB, ZA, ZB, QA, QB = util.get_dimer_data(f'./data/{args.data}.pkl', pad_dim)

    
    print("(A) Predicting...")
    qA_preds = []
    muA_preds = []
    Q_iiA_preds = []
    Q_ijA_preds = []
    for model in models:
        #print(model.summary())
        print("    Predicting on a model...")
        yA_pred = model.predict([RA, ZA, QA])
        qA_preds.append(yA_pred[0])
        muA_preds.append(yA_pred[1])
        Q_iiA_preds.append(yA_pred[2])
        Q_ijA_preds.append(yA_pred[3])
    
    print("(A) Averaging Predictions...")
    qA_avg = np.expand_dims(np.average(np.array(qA_preds), axis=0), axis=-1)
    muA_avg = np.average(np.array(muA_preds), axis=0)
    Q_iiA_avg = np.average(np.array(Q_iiA_preds), axis=0)
    Q_ijA_avg = np.average(np.array(Q_ijA_preds), axis=0)

    print(qA_avg.shape)
    print(muA_avg.shape)
    print(Q_iiA_avg.shape)
    print(Q_ijA_avg.shape)
    avgA = np.concatenate([qA_avg, muA_avg, Q_iiA_avg, Q_ijA_avg], axis=-1)
    #avgA = np.transpose(avgA, [1,0,2,3])
    print(avgA.shape)
    
    print("(A) Saving...")
    df = pd.read_pickle(f"./data/{args.data}.pkl")
    listA1 = list(avgA)
    listA2 = df.multipoles_A.tolist()

    assert len(listA1) == len(listA2)
    for ai in range(len(listA1)):
        assert listA1[ai].shape == listA2[ai].shape
        print(np.max(np.abs(listA1[ai] - listA2[ai])))
    #print(df.multipoles_A.tolist())

    #df['multipoles_A'] = list(avgA)
    #df.to_pickle(f"./data/{args.dataset_test}.pkl")
    del df


    print("(B) Predicting...")
    qB_preds = []
    muB_preds = []
    Q_iiB_preds = []
    Q_ijB_preds = []
    for model in models:
        #print(model.summary())
        print("    Predicting on a model...")
        yB_pred = model.predict([RB, ZB, QB])
        qB_preds.append(yB_pred[0])
        muB_preds.append(yB_pred[1])
        Q_iiB_preds.append(yB_pred[2])
        Q_ijB_preds.append(yB_pred[3])
    
    print("(B) Averaging Predictions...")
    qB_avg = np.expand_dims(np.average(np.array(qB_preds), axis=0), axis=-1)
    muB_avg = np.average(np.array(muB_preds), axis=0)
    Q_iiB_avg = np.average(np.array(Q_iiB_preds), axis=0)
    Q_ijB_avg = np.average(np.array(Q_ijB_preds), axis=0)
    
    avgB = np.concatenate([qB_avg, muB_avg, Q_iiB_avg, Q_ijB_avg], axis=-1)
    
    print("(B) Saving...")
    df = pd.read_pickle(f"./data/{args.dataset_test}.pkl")

    listB1 = list(avgB)
    listB2 = df.multipoles_B.tolist()

    assert len(listB1) == len(listB2)
    for bi in range(len(listB1)):
        assert listB1[bi].shape == listB2[bi].shape
        print(np.max(np.abs(listB1[bi] - listB2[bi])))
    del df
