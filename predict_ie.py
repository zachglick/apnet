import time
t_start = time.time()
print('Loading Libraries...')

import os, sys, argparse
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
#tf.keras.backend.set_learning_phase(0)

import multipoles
import util2
import util
print(f'... Done in {time.time() - t_start:.1f} seconds\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Use a trained model to predict all four SAPT0 components')

    #parser.add_argument('data',
    #                    help='Dataset for training')
    parser.add_argument('data',
                        help='Dataset for testing')

    #  model name
    parser.add_argument('pair_modelname',
                        help='Existing model to use for inference.')

    # optional arguments: feature hyperparameters
    # (used to ensure correct model reconstruction, these ought to be serialized and saved w the model)
    parser.add_argument('--acsf_nmu',
                        help='ACSF hyperparameter (number of radial gaussians).',
                        type=int,
                        default=43)
    parser.add_argument('--apsf_nmu',
                        help='APSF hyperparameter (number of angular gaussians).',
                        type=int,
                        default=21)
    parser.add_argument('--acsf_eta',
                        help='ACSF hyperparameter (radial gaussian width).',
                        type=int,
                        default=100)
    parser.add_argument('--apsf_eta',
                        help='APSF hyperparameter (angular gaussian width).',
                        type=int,
                        default=25)

    parser.add_argument('--batch_size',
                        help='Batch size for inference (number of dimers)',
                        type=int,
                        default=8)

    args = parser.parse_args(sys.argv[1:])

    data = args.data
    pair_modelname = args.pair_modelname
    batch_size = args.batch_size
    ACSF_nmu = args.acsf_nmu
    APSF_nmu = args.apsf_nmu
    ACSF_eta = args.acsf_eta
    APSF_eta = args.apsf_eta

    pad_dim = 26
    nelem = 36
    
    # this is up to the user
    nembed = 10
    nnodes = [256,128,64]
    #nnodes = [64,32,16]
    nmessage = 3

    # make the model 
    nrbf= 43
    mus = np.linspace(0.8, 5.0, nrbf)
    etas = np.array([-100.0] * nrbf)

    # ensure valid model path
    pair_modelpath = f'./pair_models/{pair_modelname}.h5'
    if not os.path.isfile(pair_modelpath):
        raise Exception(f'No model exists at {pair_modelpath}')

    atom_modelnames = ['hfadz1', 'hfadz2', 'hfadz3']
    for atom_modelname in atom_modelnames:
        if not os.path.isfile(f'atom_models/{atom_modelname}.hdf5'):
            raise Exception(f'No model exists at atom_models/{atom_modelname}.hdf5')

    print("Loading Models...")
    t_start = time.time()
    # Initialize model and load weights from file for testing
    model = util2.make_model(nZ=nembed)
    model.load_weights(f'{pair_modelpath}')
    
    atom_models = []
    for atom_modelname in atom_modelnames:
        atom_models.append(util.get_model(mus, etas, pad_dim, nelem, nembed, nnodes, nmessage))
        atom_models[-1].load_weights(f'./atom_models/{atom_modelname}.hdf5')
        #converter = tf.lite.TFLiteConverter.from_keras_model(atom_models[-1])
        #atom_models[-1] = converter.convert()
        #atom_models[-1].call = tf.function(atom_models[-1].call, experimental_relax_shapes=True)
        #print(atom_models[-1].layers)

    print(f'... Done in {time.time() - t_start:.1f} seconds\n')

    # load test data
    print("Loading Data...")
    t_start = time.time()

    # dimer is a list of tuples
    # dimer[i][0] = RA [nA x 3]
    # dimer[i][1] = RB [nB x 3]
    # dimer[i][2] = ZA [nA]
    # dimer[i][3] = ZB [nB]

    dimer, label = util2.get_dimers(data)
    N = len(dimer)

    #print(f'  Batch size:  {batch_size}')
    #print(f'  ACSF count:  {ACSF_nmu}')
    #print(f'  ACSF eta:    {ACSF_eta}')
    #print(f'  APSF count:  {APSF_nmu}')
    #print(f'  APSF eta:    {APSF_eta}')
    #print(f'  Dimer count: {N}')

    print(f'... Done in {time.time() - t_start:.1f} seconds\n')

    feature_time = 0.0
    yt_preds = []
    yt_errs = []

    print('Calculating multipole electrostatics...')
    t_start = time.time()

    for i, d in enumerate(dimer):

        nA, nB = len(d[0]), len(d[1])

        yA_preds = []
        yB_preds = []

        RAti = np.zeros((1, pad_dim, 3))
        RBti = np.zeros((1, pad_dim, 3))
        RAti[0,:nA,:] = d[0] 
        RBti[0,:nB,:] = d[1] 

        ZAti = np.zeros((1, pad_dim))
        ZBti = np.zeros((1, pad_dim))
        ZAti[0,:nA] = d[2] 
        ZBti[0,:nB] = d[3] 

        aQAti = np.zeros((1, pad_dim, pad_dim, 1))
        aQBti = np.zeros((1, pad_dim, pad_dim, 1))
        aQAti[0,:nA,:nA,0] = d[4]
        aQBti[0,:nB,:nB,0] = d[5]
        
        for atom_model in atom_models:

            yA_pred = atom_model.predict_on_batch([RAti, ZAti, aQAti])
            yB_pred = atom_model.predict_on_batch([RBti, ZBti, aQBti])

            yA_pred = np.concatenate([np.expand_dims(yA_pred[0], axis=-1), yA_pred[1], yA_pred[2], yA_pred[3]], axis=-1)
            yA_preds.append(yA_pred)

            yB_pred = np.concatenate([np.expand_dims(yB_pred[0], axis=-1), yB_pred[1], yB_pred[2], yB_pred[3]], axis=-1)
            yB_preds.append(yB_pred)

        QAti = np.average(yA_preds, axis=0)
        QBti = np.average(yB_preds, axis=0)

        QAti = QAti[0,:nA,:]
        QBti = QBti[0,:nB,:]

        elst_mtp, pair_mtp = multipoles.eval_dimer(d[0], d[1], d[2], d[3], QAti, QBti)
        dimer[i] = (d[0], d[1], d[2], d[3], QAti, QBti, pair_mtp)

        features = util2.make_features(*dimer[i])

        # these are pairwise
        sapt_pred = util2.predict_single(model, features)
        sapt_pred = np.sum(sapt_pred, axis=0)
        yt_preds.append(sapt_pred)
        yt_errs.append(sapt_pred - label[i])

    #print(f'... Done in {time.time() - t_start:.1f} seconds\n')

    #print('Predicting Interaction Energies...')
    #t_start = time.time()
    #dimer_batches, label_batches = util2.make_batches(dimer, label, batch_size)
    #for dimer_batch, label_batch in zip(dimer_batches, label_batches):
    #    feature_start = time.time()
    #    feature_batch = [util2.make_features(*d) for d in dimer_batch]
    #    feature_time += (time.time() - feature_start)
    #    for ft, lt in zip(feature_batch, label_batch):
    #        yt_pred = util2.predict_single(model, ft)
    #        yt_pred = np.sum(yt_pred, axis=0)
    #        yt_preds.append(yt_pred)
    #        yt_errs.append(yt_pred - lt)
    #print(f'... Features took {feature_time:.1f} seconds')
    print(f'... Done in {time.time() - t_start:.1f} seconds\n')

    yt_errs = np.concatenate(yt_errs).reshape(-1,4)
    yt_preds = np.concatenate(yt_preds).reshape(-1,4)
    yt_maes = np.average(np.absolute(yt_errs), axis=0)

    np.set_printoptions(suppress=True)
    #print('preds')
    #print(yt_preds)
    #print('labs')
    #print(label)
    #print('errs')
    #print(yt_errs)
    #print('maes')
    print(yt_maes)
