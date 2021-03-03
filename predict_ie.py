import time
t_start = time.time()
print('Loading Libraries...')

import os, sys, argparse
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

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
    nembed = 10
    nelem = 36
    
    # this is up to the user
    nembed = 10
    nnodes = [256,128,64]
    nmessage = 3

    # make the model 
    mus = np.linspace(0.8, 5.0, 43)
    etas = np.array([-100.0] * 43)

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
        #atom_models[-1].call = tf.function(atom_models[-1].call, experimental_relax_shapes=True)
        print(atom_models[-1].layers)

    print(f'... Done in {time.time() - t_start:.1f} seconds\n')

    # load test data
    print("Loading Data...")
    t_start = time.time()
    dimer, label = util2.get_dimers(data)

    N = len(dimer)

    #print(f'  Batch size:  {batch_size}')
    #print(f'  ACSF count:  {ACSF_nmu}')
    #print(f'  ACSF eta:    {ACSF_eta}')
    #print(f'  APSF count:  {APSF_nmu}')
    #print(f'  APSF eta:    {APSF_eta}')
    #print(f'  Dimer count: {N}')

    RAt, RBt, ZAt, ZBt, aQAt, aQBt, QAt_ref, QBt_ref = util2.pad_molecules(dimer, pad_dim)
    print(f'... Done in {time.time() - t_start:.1f} seconds\n')

    print('Predicting Multipoles...')
    t_start = time.time()
    qA_preds = []
    muA_preds = []
    Q_iiA_preds = []
    Q_ijA_preds = []

    qB_preds = []
    muB_preds = []
    Q_iiB_preds = []
    Q_ijB_preds = []

    for atom_model in atom_models:

        #yA_pred = util2.predict_single(atom_model, [RAt, ZAt, aQAt])
        #yA_pred = atom_model([RAt, ZAt, aQAt], training=False)
        yA_pred = atom_model.predict([RAt, ZAt, aQAt], batch_size=4)
        #yA_pred = atom_model.predict_on_batch([RAt, ZAt, aQAt])#batch_size=1)

        qA_preds.append(yA_pred[0])
        muA_preds.append(yA_pred[1])
        Q_iiA_preds.append(yA_pred[2])
        Q_ijA_preds.append(yA_pred[3])

        #yB_pred = util2.predict_single(atom_model, [RBt, ZBt, aQBt])
        #yB_pred = atom_model([RBt, ZBt, aQBt], training=False)
        yB_pred = atom_model.predict([RBt, ZBt, aQBt], batch_size=4)
        #yB_pred = atom_model.predict_on_batch([RBt, ZBt, aQBt])#batch_size=1)

        qB_preds.append(yB_pred[0])
        muB_preds.append(yB_pred[1])
        Q_iiB_preds.append(yB_pred[2])
        Q_ijB_preds.append(yB_pred[3])




    

        #qB_pred = []
        #muB_pred = []
        #Q_iiB_pred = []
        #Q_ijB_pred = []

        #for ib in range(len(RBt)):
        #    yB_pred = atom_model([RBt[[ib]], ZBt[[ib]], aQBt[[ib]]], training=False)
        #    qB_pred.append(yB_pred[0])
        #    muB_pred.append(yB_pred[1])
        #    Q_iiB_pred.append(yB_pred[2])
        #    Q_ijB_pred.append(yB_pred[3])

        #qB_preds.append(np.concatenate(qB_pred, axis=0))
        #muB_preds.append(np.concatenate(muB_pred, axis=0))
        #Q_iiB_preds.append(np.concatenate(Q_iiB_pred, axis=0))
        #Q_ijB_preds.append(np.concatenate(Q_ijB_pred, axis=0))
    
    qA_avg = np.expand_dims(np.average(np.array(qA_preds), axis=0), axis=-1)
    muA_avg = np.average(np.array(muA_preds), axis=0)
    Q_iiA_avg = np.average(np.array(Q_iiA_preds), axis=0)
    Q_ijA_avg = np.average(np.array(Q_ijA_preds), axis=0)
    QAt = np.concatenate([qA_avg, muA_avg, Q_iiA_avg, Q_ijA_avg], axis=-1)

    qB_avg = np.expand_dims(np.average(np.array(qB_preds), axis=0), axis=-1)
    muB_avg = np.average(np.array(muB_preds), axis=0)
    Q_iiB_avg = np.average(np.array(Q_iiB_preds), axis=0)
    Q_ijB_avg = np.average(np.array(Q_ijB_preds), axis=0)
    QBt = np.concatenate([qB_avg, muB_avg, Q_iiB_avg, Q_ijB_avg], axis=-1)
    print(f'... Done in {time.time() - t_start:.1f} seconds\n')

    print('Calculating multipole electrostatics...')
    t_start = time.time()
    elst_mtp_t = []
    for i in range(N):
        elst_mtp, pair_mtp = multipoles.eval_dimer(RAt[i], RBt[i], ZAt[i], ZBt[i], QAt[i], QBt[i])
        elst_mtp_t.append(pair_mtp)
    print(f'... Done in {time.time() - t_start:.1f} seconds\n')

    for i, d in enumerate(dimer):
        nA, nB = len(d[0]), len(d[1])
        dimer[i] = (d[0], d[1], d[2], d[3], QAt[i,:nA,:], QBt[i,:nB,:], elst_mtp_t[i]) # charge only

    feature_time = 0.0
    yt_preds = []
    yt_errs = [] # record testing error

    print('Predicting Interaction Energies...')
    t_start = time.time()
    dimer_batches, label_batches = util2.make_batches(dimer, label, batch_size)
    for dimer_batch, label_batch in zip(dimer_batches, label_batches):
        feature_start = time.time()
        feature_batch = [util2.make_features(*d) for d in dimer_batch]
        feature_time += (time.time() - feature_start)
        for ft, lt in zip(feature_batch, label_batch):
            yt_pred = util2.predict_single(model, ft)
            yt_pred = np.sum(yt_pred, axis=0)
            yt_preds.append(yt_pred)
            yt_errs.append(yt_pred - lt)
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
