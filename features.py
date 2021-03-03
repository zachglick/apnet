import sys
import time
from multiprocessing import Pool
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from scipy.spatial import distance_matrix

zlist = [1.0, 6.0, 7.0, 8.0, 9.0, 11.0, 15.0, 16.0, 17.0, 35.0]
charge_to_index = { 0.0  : 100000,
                    1.0  : 0,
                    6.0  : 1,
                    7.0  : 2,
                    8.0  : 3,
                    9.0  : 4,
                    11.0 : 5,
                    15.0 : 6,
                    16.0 : 7,
                    17.0 : 8,
                    35.0 : 9,
                   }


def cutoff(D, rc1=0.0, rc2=8.0):
    """
    Cutoff function used to ensure locality of features.

    Args:
        D: distance matrix with dimensions [N x N]
        rc1 : min cutoff distance
        rc2 : max cutoff distance
    Returns: 
        cutoff matrix with same shape as D
    """
    C = (np.cos(np.pi * (D - rc1) / (rc2 - rc1)) + 1.0) / 2.0
    C[D >= rc2] = 0.0
    C[D <= rc1] = 1.0
    np.fill_diagonal(C, 0.0)
    return C


def cos_theta(R):
    """ 
    Calculates the cosine between all possible angles given a set of points.

    Args:
        R: cartesian coordinates of a monomer or dimer [NATOM x 3]

    Returns three index tensor ct, where ct[i,j,k] is cosine(theta_jik).
    The first index (i) is the center point of the angle.
    i, j, and k index atoms in R.
    """
    
    ct = np.zeros((R.shape[0], R.shape[0], R.shape[0]))
    for i in range(R.shape[0]):
        dRxyz = R - R[i]
        dR1 = np.linalg.norm(dRxyz, axis=1)
        num = np.inner(dRxyz, dRxyz)
        denom = np.outer(dR1, dR1)
        ct[i,:,:] = (num / denom)
    return np.nan_to_num(ct)


def cos_theta_im(RA, RB):
    """ 
    Calculates the cosine between all possible intermolecular angles given set of points in two monomers.

    Args:
        RA: cartesian coordinates of monomer A
        RB: cartesian coordinates of monomer B

    Returns three index tensor ct, where ct[i,j,k] is cosine(theta_jik).
    The first index (i) is the center point of the angle.
    i and k index atoms in RA while j indexes atoms in RB.
    """
    
    ct = np.zeros((RA.shape[0], RB.shape[0], RA.shape[0]))
    for i in range(RA.shape[0]):
        dRAxyz = RA - RA[i]
        dRBxyz = RB - RA[i]

        dRA1 = np.linalg.norm(dRAxyz, axis=1)
        dRB1 = np.linalg.norm(dRBxyz, axis=1)

        num = np.inner(dRBxyz, dRAxyz)
        denom = np.outer(dRB1, dRA1)

        ct[i,:,:] = (num / denom)

    return np.nan_to_num(ct)
        

def acsfs(Z, R, mus, eta=100.0, rc1=0.0, rc2=8.0):
    """
    Calculates the radial atom centered symmetry functions of all atoms in a monomer

    Args:
        Z: atom type array with dimensions [NATOM]
        R: atom coordinate array with dimensions [NATOM x 3]
        mus: ACSF descriptor shift parameter array with dimensions [NMU]
        eta: ACSF descriptor gaussian width parameter
        rc1: cutoff function first parameter
        rc2: cutoff function second parameter

    Returns a tensor containing the radial ACSFs for all atoms in a monomer with dimensions [NATOM x NMU X NZ]
    """

    natom = len(R)
    nmu = len(mus)
    ntype = len(zlist)

    zindex = [charge_to_index[z] for z in Z]

    Dxyz = distance_matrix(R, R)
    C = cutoff(Dxyz, rc1, rc2)
    G = np.zeros((natom, nmu, ntype))

    for ind1 in range(natom):
        Dxyz_atom = Dxyz[ind1]
        for ind2, d in enumerate(Dxyz_atom):
            if d >= rc2 or zindex[ind1] > 100 or zindex[ind2] > 100:
                continue
            G[ind1,:,zindex[ind2]] += np.exp(-1.0 * np.square(d - mus) * eta) * C[ind1, ind2]

    return G


def apsfs(ZA, RA, ZB, RB, mus, eta=100.0, rc1=0.0, rc2=8.0):
    """
    Calculates the angular atom pair symmetry functions of all pairs atoms in two monomers

    Args:
        ZA: monomer A atom type array with dimensions [NATOMA]
        RA: monomer A atom coordinate array with dimensions [NATOM x 3]
        ZB: monomer B atom type array with dimensions [NATOMB]
        RB: monomer B atom coordinate array with dimensions [NATOM x 3]
        mus: APSF descriptor shift parameter array with dimensions [NMU]
        eta: APSF descriptor gaussian width parameter
        rc1: cutoff function first parameter
        rc2: cutoff function second parameter

    Returns a tensor containing the radial ACSFs for all atoms in a monomer with dimensions [NATOM x NMU X NZ]
    """

    natomA = len(RA)
    natomB = len(RB)
    nmu = len(mus)
    ntype = len(zlist)

    zindexA = [charge_to_index[z] for z in ZA]
    zindexB = [charge_to_index[z] for z in ZB]

    DAA = distance_matrix(RA, RA)
    CAA = cutoff(DAA, rc1, rc2)
    DAB = distance_matrix(RA, RB)

    A = cos_theta_im(RA, RB) # angles

    if natomA > 50 or natomB > 50:
        # sparse implementation, good for big systems
        G = np.zeros((natomA, natomB, nmu, ntype))
        for a_A in range(natomA):
            e_A = zindexA[a_A]
            if e_A > 100:
                continue
            for a_B, d_AB in enumerate(DAB[a_A]):
                e_B = zindexB[a_B]
                if d_AB > rc2 or e_B > 100:
                    continue
                for a_A2, d_AA2 in enumerate(DAA[a_A]):
                    e_A2 = zindexA[a_A2]
                    if d_AA2 > rc2 or e_A2 > 100:
                        continue
                    c_th = A[a_A, a_B, a_A2]
                    G[a_A, a_B, :, e_A2] += np.exp(-1.0 * np.square(c_th - mus) * eta) * CAA[a_A, a_A2] #* C[a_j, a_k]

    else:
        # dense implementation, good for small systems
        G = np.expand_dims(A, 2)
        G = np.tile(G, [1,1,nmu,1]) - mus.reshape((1,1,nmu,1))
        G = np.exp(-1.0 * np.square(G) * eta)
        G = np.einsum('ijkl,il->ijkl',G,CAA)
        emaskA = np.zeros((natomA, ntype))
        for a, e in enumerate(zindexA):
            emaskA[a,e] = 1
        G = np.einsum('ijkl,lz->ijkz', G, emaskA)
        G = np.einsum('ijkl,ij->ijkl', G, (DAB < rc2))

    return G


#def asyms(Z, R, mus, eta=100.0, rc1=0.0, rc2=8.0):
#    """
#    Calculates the angular atom centered symmetry functions of all atoms in a monomer
#    """
#
#    natom = len(R)
#    nmu = len(mus)
#    ntype = len(zlist)
#
#    zindex = [charge_to_index[z] for z in Z]
#
#    D = distance_matrix(R, R)
#    C = cutoff(D, rc1, rc2)
#    A = cos_theta(R)
#    G = np.zeros((natom, nmu, ntype, ntype))
#
#    for a_j in range(natom):
#        e_j = zindex[a_j]
#        if e_j > 100:
#            continue
#        for a_i, d_i in enumerate(D[a_j]):
#            e_i = zindex[a_i]
#            if d_i > rc2 or e_i > 100:
#                continue
#            for a_k, d_k in enumerate(D[a_j]):
#                e_k = zindex[a_k]
#                if d_k > rc2 or e_k > 100:
#                    continue
#                c_th = A[a_j, a_i, a_k]
#                G[a_j,:,e_i,e_k] += np.exp(-1.0 * np.square(c_th - mus) * eta) * C[a_j, a_i] * C[a_j, a_k]
#                if e_k != e_i:
#                    G[a_j,:,e_k,e_i] += np.exp(-1.0 * np.square(c_th - mus) * eta) * C[a_j, a_k] * C[a_j, a_i]
#    return G * 0.5

def calculate_dimer(RA, RB, ZA, ZB, ACSF_nmu=43, APSF_nmu=21, ACSF_eta=100, APSF_eta=25):

    ACSF_mus = np.linspace(0.8, 5.0, ACSF_nmu)
    APSF_mus = np.linspace(-1.0, 1.0, APSF_nmu)

    ACSF_A = acsfs(ZA, RA, ACSF_mus, ACSF_eta, 0.0, 8.0)
    ACSF_B = acsfs(ZB, RB, ACSF_mus, ACSF_eta, 0.0, 8.0)
    APSF_A_B = apsfs(ZA, RA, ZB, RB, APSF_mus, APSF_eta, 0.0, 8.0)
    APSF_B_A = np.transpose(apsfs(ZB, RB, ZA, RA, APSF_mus, APSF_eta, 0.0, 8.0), axes=[1,0,2,3])

    return ACSF_A, ACSF_B, APSF_A_B, APSF_B_A

def calculate(dataset, ACSF_nmu=43, APSF_nmu=21, ACSF_eta=100, APSF_eta=25):
    """
    Calculate and save ACSF and APSF features for a given dataset
    """

    df_dimers = pd.read_pickle(f'datasets/{dataset}/dimers.pkl')
    N_dimer = len(df_dimers.index)

    # get coordinates and atom types, catch common errors
    try:
        df_ACSF = df_dimers[['RA', 'RB', 'ZA', 'ZB']].copy(deep=True)
        df_APSF = df_dimers[['RA', 'RB', 'ZA', 'ZB']].copy(deep=True)
    except KeyError:
        print('Error: Dataset needs fields "RA", "RB", "ZA", and "ZB" for every dimer\n')
        raise

    # TODO: experiment with these ranges
    ACSF_mus = np.linspace(0.8, 5.0, ACSF_nmu)
    APSF_mus = np.linspace(-1.0, 1.0, APSF_nmu)

    ACSF_A = []
    ACSF_B = []
    APSF_A_B = []
    APSF_B_A = []

    counter = 0
    start = time.time()
    for index, row in df_dimers.iterrows():

        RA = row['RA']
        RB = row['RB']
        ZA = row['ZA']
        ZB = row['ZB']

        ACSF_A.append(acsfs(ZA, RA, ACSF_mus, ACSF_eta, 0.0, 8.0))
        ACSF_B.append(acsfs(ZB, RB, ACSF_mus, ACSF_eta, 0.0, 8.0))
        APSF_A_B.append(apsfs(ZA, RA, ZB, RB, APSF_mus, APSF_eta, 0.0, 8.0))
        APSF_B_A.append(np.transpose(apsfs(ZB, RB, ZA, RA, APSF_mus, APSF_eta, 0.0, 8.0), axes=[1,0,2,3]))

        if (counter+1) % 500 == 0:
            dt = time.time() - start
            rate = (counter + 1) / dt
            print(f'{(counter+1):5d} of {N_dimer:5d} dimers done in {int(dt):6d} s ({int(rate)} dimers / s)')
        counter += 1

    df_ACSF['ACSF_A'] = ACSF_A
    df_ACSF['ACSF_B'] = ACSF_B
    df_APSF['APSF_A_B'] = APSF_A_B
    df_APSF['APSF_B_A'] = APSF_B_A

    df_ACSF.to_pickle(f'datasets/{dataset}/ACSF_{ACSF_nmu}_{ACSF_eta}.pkl')
    df_APSF.to_pickle(f'datasets/{dataset}/APSF_{APSF_nmu}_{APSF_eta}.pkl')

if __name__ == '__main__':
        
    df_dimers = pd.read_pickle(f'datasets/nma-training/dimers.pkl')
    N_dimer = len(df_dimers.index)
    ACSF_nmu=43
    APSF_nmu=21
    ACSF_eta=100
    APSF_eta=25

    data_dimers = list(zip(df_dimers['RA'], df_dimers['RB'], df_dimers['ZA'], df_dimers['ZB']))

    start = time.time()
    #for i, datum in enumerate(data_dimers):
    #    #if ((i + 1) % 50) == 0:
    #    #    dt = time.time() - start
    #    #    rate = (i + 1) / dt
    #    #    print(f'{(i+1):5d} of {N_dimer:5d} dimers done in {int(dt):6d} s ({int(rate)} dimers / s)')
    #    calculate_dimer(*datum)
    #print(time.time() - start)

    start = time.time()
    with Pool(processes=6) as pool:
        feats = pool.starmap(calculate_dimer, data_dimers)
    print(time.time() - start)
