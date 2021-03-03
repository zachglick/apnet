import os, sys, argparse, math, time
from multiprocessing import Pool
import numpy as np
from scipy.spatial import distance_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

import multipoles
import util2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Use a trained model to predict all four SAPT0 components')

    #parser.add_argument('data',
    #                    help='Dataset for training')
    parser.add_argument('data',
                        help='Dataset for testing')

    #  model name
    parser.add_argument('modelname',
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

    batch_size = args.batch_size
    ACSF_nmu = args.acsf_nmu
    APSF_nmu = args.apsf_nmu
    ACSF_eta = args.acsf_eta
    APSF_eta = args.apsf_eta

    # ensure valid model path
    model_path = f'./pair_models/{args.modelname}.h5'
    if not os.path.isfile(model_path):
        raise Exception(f'No model exists at {model_path}')

    # load test data
    dimer, label = util2.get_dimers(args.data)

    Nt = len(dimer)
    Nb = math.ceil(Nt / batch_size)

    print(f'  Batch size: {batch_size}')
    print(f'  ACSF count: {ACSF_nmu}')
    print(f'  ACSF eta:   {ACSF_eta}')
    print(f'  APSF count: {APSF_nmu}')
    print(f'  APSF eta:   {APSF_eta}')
    print(f'  Test count: {Nt}')

    # these are defined by the dataset
    pad_dim = 40
    nelem = 36

    # these are defined by the trained model, should be put into a separate save file and loaded directly
    nembed = 10
    nnodes = [256,128,64]
    nmessage = 3
    mus = np.linspace(0.8, 5.0, 43)
    etas = np.array([-100.0] * 43)

    RAt, RBt, ZAt, ZBt, QAt, QBt = util2.pad_molecules(dimer, 40)

    #print('Calculating multipoles...')
    #t_mtp = time.time()
    
    #TQt = np.zeros((Nt, pad_dim, pad_dim, 1))
    #TQv = np.zeros((Nv, pad_dim, pad_dim, 1))

    #QAt = mpnn.predict([RAt, ZAt, TQt], batch_size=1)
    #QBt = mpnn.predict([RBt, ZBt, TQt], batch_size=1)
    #QAv = mpnn.predict([RAv, ZAv, TQv], batch_size=1)
    #QBv = mpnn.predict([RBv, ZBv, TQv], batch_size=1)

    #QAt[0] = np.expand_dims(QAt[0], axis=-1)
    #QBt[0] = np.expand_dims(QBt[0], axis=-1)
    #QAv[0] = np.expand_dims(QAv[0], axis=-1)
    #QBv[0] = np.expand_dims(QBv[0], axis=-1)

    #QAt = np.concatenate(QAt, axis=-1)
    #QBt = np.concatenate(QBt, axis=-1)
    #QAv = np.concatenate(QAv, axis=-1)
    #QBv = np.concatenate(QBv, axis=-1)

    #print('QAt:', QAt.shape)

    #print(f'... Done in {int(time.time() - t_mtp)} seconds')

    print('Calculating multipole electrostatics...')
    t_mtp = time.time()
    elst_sapt_l = []
    elst_mtp_t = []
    elst_mtp_l = []
    r_cc = []
    for i in range(Nt):
        elst_mtp, pair_mtp = multipoles.eval_dimer(RAt[i], RBt[i], ZAt[i], ZBt[i], QAt[i], QBt[i])
        elst_sapt_l.append(label[i][0])
        elst_mtp_l.append(elst_mtp)
        elst_mtp_t.append(pair_mtp)
        #label[i][0] -= elst_mtp
        D = distance_matrix(RAt[i][ZAt[i] > 0], RBt[i][ZBt[i] > 0])
        r_cc.append(np.min(D))
    print(f'... Done in {int(time.time() - t_mtp)} seconds')
    np.save('elst_sapt_l.npy', elst_sapt_l)
    np.save('elst_mtp_l.npy', elst_mtp_l)
    np.save('r_cc.npy', r_cc)

    for i, d in enumerate(dimer):
        nA, nB = len(d[0]), len(d[1])
        dimer[i] = (d[0], d[1], d[2], d[3], QAt[i,:nA,:], QBt[i,:nB,:], elst_mtp_t[i]) # charge only

    # Initialize model and load weights from file for testing
    model = util2.make_model(nZ=10)
    model.load_weights(f'{model_path}')

    feature_time = 0.0
    start = time.time()

    yt_preds = []
    yt_errs = [] # record testing error

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


    yt_errs = np.concatenate(yt_errs)
    yt_preds = np.concatenate(yt_preds)
    yt_maes = np.average(np.absolute(yt_errs), axis=0)

    np.save(f'test_preds.npy', np.array(yt_preds))
    np.save(f'test_errs.npy', np.array(yt_errs))

    np.set_printoptions(suppress=True)
    print('preds')
    print(yt_preds.reshape(-1,4))
    print('labs')
    print(label)
    print('errs')
    print(yt_errs.reshape(-1,4))
