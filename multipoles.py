import math
import numpy as np
import h5py

elem_to_z = { 
    'H'  : 1,
    'B'  : 3,
    'C'  : 6,
    'N'  : 7,
    'O'  : 8,
    'F'  : 9,
    'NA' : 11,
    'P'  : 15, 
    'S'  : 16, 
    'CL' : 17, 
    'SE' : 34, 
    'BR' : 35, 
}

def proc_molden(name):
    """ Get coordinates (a.u.) and atom types from a molden.
    Accounts for ghost atoms """

    with open(name, 'r') as fp: 
        data = fp.read().split('[GTO]')[0].strip()
    data = data.split('\n')[2:]
    data = [line.strip().split() for line in data]
    Z = [line[0] for line in data]

    try:
        Z = [elem_to_z[elem] for elem in Z]
    except:
        print(name)
        return 0, 0
    Z = np.array(Z, dtype=np.int64)

    R = [line[3:] for line in data]
    R = [[float(xyz) for xyz in line] for line in R]
    R = np.array(R, dtype=np.float64)

    mask = [line[2] for line in data]
    mask = [(d != '0') for d in mask]

    return R[mask], Z[mask]

def qpole_redundant(unique):

    assert unique.shape == (6,)
    redundant = np.zeros((3,3))

    redundant[0,0] = unique[0]
    redundant[0,1] = unique[1]
    redundant[1,0] = unique[1]
    redundant[0,2] = unique[2]
    redundant[2,0] = unique[2]
    redundant[1,1] = unique[3]
    redundant[1,2] = unique[4]
    redundant[2,1] = unique[4]
    redundant[2,2] = unique[5]
    return redundant


def T_cart(RA, RB):

    dR = RB - RA
    R = np.linalg.norm(dR)

    delta = np.identity(3)

    T0 = (R ** -1)
    T1 = (R ** -3) * (-1.0 * dR)
    T2 = (R ** -5) * (3 * np.outer(dR, dR) - R * R * delta)

    Rdd = np.multiply.outer(dR, delta)
    T3 = (R ** -7) * -1.0 * (  15 * np.multiply.outer(np.outer(dR, dR), dR) 
                             - 3  * R * R * (Rdd + Rdd.transpose(1,0,2) + Rdd.transpose(2,0,1)))

    RRdd = np.multiply.outer(np.outer(dR, dR), delta)
    dddd = np.multiply.outer(delta, delta)
    T4 = (R ** -9) * (  105 * np.multiply.outer( np.outer(dR, dR), np.outer(dR, dR) )
                      - 15 * R * R * (RRdd + RRdd.transpose(0,2,1,3) + RRdd.transpose(0,3,2,1) + RRdd.transpose(2,1,0,3) + RRdd.transpose(3,1,2,0) + RRdd.transpose(2,3,0,1))
                      + 3 * (R ** 4) * (dddd + dddd.transpose(0,2,1,3) + dddd.transpose(0,3,2,1)))

    return T0, T1, T2, T3, T4

def eval_interaction(RA, qA, muA, thetaA, RB, qB, muB, thetaB):

    T0, T1, T2, T3, T4 = T_cart(RA, RB)
    
    # Trace is already taken care of
    #if False:
    #    traceA = np.trace(thetaA)
    #    thetaA[0,0] -= traceA / 3.0
    #    thetaA[1,1] -= traceA / 3.0
    #    thetaA[2,2] -= traceA / 3.0
    #    traceB = np.trace(thetaB)
    #    thetaB[0,0] -= traceB / 3.0
    #    thetaB[1,1] -= traceB / 3.0
    #    thetaB[2,2] -= traceB / 3.0

    E_qq = np.sum(T0 * qA * qB)
    E_qu = np.sum(T1 * (qA * muB - qB * muA))
    E_qQ = np.sum(T2 * (qA * thetaB + qB * thetaA)) * (1.0 / 3.0)
 
    E_uu = np.sum(T2 * np.outer(muA, muB)) * (-1.0)
    E_uQ = np.sum(T3 * (np.multiply.outer(muA, thetaB) - np.multiply.outer(muB, thetaA))) * (-1.0 / 3.0) # sign ??

    E_QQ = np.sum(T4 * np.multiply.outer(thetaA, thetaB)) * (1.0 / 9.0) 

    # partial-charge electrostatic energy
    E_q = E_qq

    # dipole correction
    E_u = E_qu + E_uu

    # quadrupole correction
    E_Q = E_qQ + E_uQ + E_QQ

    return E_q + E_u + E_Q


def eval_dimer2(RA, RB, ZA, ZB, QA, QB):

    maskA = (ZA >= 1)
    maskB = (ZB >= 1)

    # Keep R in a.u. (molden convention)
    RA_temp = RA[maskA] * 1.88973
    RB_temp = RB[maskB] * 1.88973
    ZA_temp = ZA[maskA]
    ZB_temp = ZB[maskB]
    QA_temp = QA[maskA]
    QB_temp = QB[maskB]

    quadrupole_A = []
    quadrupole_B = []

    for ia in range(len(RA_temp)):
        #QA_temp[ia][4:10] = (3.0/2.0) * qpole_redundant(QA_temp[ia][4:10])
        quadrupole_A.append((3.0/2.0) * qpole_redundant(QA_temp[ia][4:10]))

    for ib in range(len(RB_temp)):
        #QB_temp[ib][4:10] = (3.0/2.0) * qpole_redundant(QB_temp[ib][4:10])
        quadrupole_B.append((3.0/2.0) * qpole_redundant(QB_temp[ib][4:10]))

    tot_energy = 0.0

    # calculate multipole electrostatics for each atom pair
    for ia in range(len(RA_temp)):
        for ib in range(len(RB_temp)):
            rA = RA_temp[ia]
            qA = QA_temp[ia]

            rB = RB_temp[ib]
            qB = QB_temp[ib]

            pair_energy = eval_interaction(rA,
                                          qA[0], 
                                          qA[1:4], 
                                          #qA[4:10], 
                                          quadrupole_A[ia],
                                          rB, 
                                          qB[0], 
                                          qB[1:4], 
                                          #qB[4:10])
                                          quadrupole_B[ib])

            tot_energy += pair_energy

    Har2Kcalmol = 627.5094737775374055927342256

    return tot_energy * Har2Kcalmol




def eval_dimer(RA, RB, ZA, ZB, QA, QB):

    #print()
    #print(RA.shape, ZA.shape, QA.shape)
    #print(RB.shape, ZB.shape, QB.shape)

    # Keep R in a.u. (molden convention)
    RA_temp = RA * 1.88973
    RB_temp = RB * 1.88973

    tot_energy = 0.0

    maskA = (ZA >= 1)
    maskB = (ZB >= 1)

    pair_mat = np.zeros((int(np.sum(maskA, axis=0)), int(np.sum(maskB, axis=0))))
    #print(pair_mat.shape)
    #QA[:,0] -= maskA * np.sum(QA[:,0]) / np.sum(maskA)
    #QB[:,0] -= maskB * np.sum(QB[:,0]) / np.sum(maskB)
    #QA[:,0] -= np.average(QA[:,0]) 
    #QB[:,0] -= np.average(QB[:,0]) 
    #print(f'{np.sum(QA[:,0]):.2f} {np.sum(QB[:,0]):.2f}')
    #print(QA[:,0], QB[:,0])

    # calculate multipole electrostatics for each atom pair
    for ia in range(len(RA_temp)):
        for ib in range(len(RB_temp)):
            rA = RA_temp[ia]
            zA = ZA[ia]
            qA = QA[ia]

            rB = RB_temp[ib]
            zB = ZB[ib]
            qB = QB[ib]

            if (zA == 0) or (zB == 0):
                continue

            pair_energy = eval_interaction(rA,
                                          qA[0], 
                                          qA[1:4], 
                                          (3.0/2.0) * qpole_redundant(qA[4:10]), 
                                          rB, 
                                          qB[0], 
                                          qB[1:4], 
                                          (3.0/2.0) * qpole_redundant(qB[4:10]))

            #print('pair', pair_energy)
            tot_energy += pair_energy
            pair_mat[ia][ib] = pair_energy

    Har2Kcalmol = 627.5094737775374055927342256

    return tot_energy * Har2Kcalmol, pair_mat * Har2Kcalmol


#def eval_dimer(moldenA, moldenB, h5A, h5B, print_=False):
#
#    # Keep R in a.u. (molden convention)
#    RA, ZA = proc_molden(moldenA)
#    RB, ZB = proc_molden(moldenB)
#
#    # Get horton output
#    hfA = h5py.File(h5A, 'r')
#    hfB = h5py.File(h5B, 'r')
#
#    # get multipoles
#    QA = hfA['cartesian_multipoles'][:]
#    QB = hfB['cartesian_multipoles'][:]
#
#    tot_energy = 0.0
#    pair_energies_l = []
#
#    # calculate multipole electrostatics for each atom pair
#    for ia in range(len(ZA)):
#        for ib in range(len(ZB)):
#            rA = RA[ia]
#            zA = ZA[ia]
#            qA = QA[ia]
#
#            rB = RB[ib]
#            zB = ZB[ib]
#            qB = QB[ib]
#
#            #print(zA, qA[0], zB, qB[0])
#
#            ## [E0 (qq), E1 (qmu), E2(qTh+mumu), E3(muTh), E4(ThTh)]
#            #pair_energies = eval_interaction(rA, qA[0], qA[1:4], (3.0/2.0) * qpole_redundant(qA[4:10]), 
#            #        rB, qB[0], qB[1:4], (3.0/2.0) * qpole_redundant(qB[4:10]))
#
#            # [E0 (qq), E1 (qmu), E2(qTh+mumu), E3(muTh), E4(ThTh)]
#            pair_energies = eval_interaction_damp(rA, zA, qA[0] - zA, qA[1:4], (3.0/2.0) * qpole_redundant(qA[4:10]), 
#                    rB, zB, qB[0] - zB, qB[1:4], (3.0/2.0) * qpole_redundant(qB[4:10]))
#            pair_energy = np.sum(pair_energies)
#            pair_energies_l.append(pair_energies)
#
#            tot_energy += pair_energy
#
#    pair_energies_l = np.array(pair_energies_l)
#    Har2Kcalmol = 627.509
#
#    if print_:
#        print('  Energies (mEh)')
#        print('     E_q        E_u        E_Q        Total')
#        for l in pair_energies_l:
#            print(f'{l[0]*Har2Kcalmol:10.5f} {l[1]*Har2Kcalmol:10.5f} {l[2]*Har2Kcalmol:10.5f} {np.sum(l)*Har2Kcalmol:10.5f}')
#        l = np.sum(pair_energies_l, axis=0)
#        print('  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#        print(f'{l[0]*Har2Kcalmol:10.5f} {l[1]*Har2Kcalmol:10.5f} {l[2]*Har2Kcalmol:10.5f} {np.sum(l)*Har2Kcalmol:10.5f}')
#
#    E_q = np.sum(pair_energies_l[:,:1])
#    E_qu = np.sum(pair_energies_l[:,:2])
#    E_quQ = np.sum(pair_energies_l[:,:3])
#    return E_q, E_qu, E_quQ


if __name__ == '__main__':

    import pandas as pd

    df = pd.read_pickle('../directional-mpnn/data/HBC6_hfjdz.pkl')
    df_prd = pd.read_pickle('../directional-mpnn/preds/HBC6_hfjdz_modelB.pkl')

    print(df.cartesian_multipoles[0].shape)
    print(df_prd.cartesian_multipoles_prd[0].shape)
    print(df.columns)

    df_elst = {
            'name':     [],
            'dimer':    [],
            'dist':     [],
            'e_mbis_q': [],
            'e_mbis_u': [],
            'e_mbis_Q': [],
            'e_mpnn_q': [],
            'e_mpnn_u': [],
            'e_mpnn_Q': [],
            'e_sapt':   [],
    }

    errs = []
    for i in range(0, len(df.index), 2):
        print(i, df.name[i])
        RA = df.R[i]
        RB = df.R[i+1]
        ZA = df.Z[i]
        ZB = df.Z[i+1]
        QA = df.cartesian_multipoles[i]
        QB = df.cartesian_multipoles[i+1]
        QA_prd = df_prd.cartesian_multipoles_prd[i]
        QB_prd = df_prd.cartesian_multipoles_prd[i+1]
        #print(QA[:,0]-QA_prd[:,0])
        #print(QB[:,0]-QB_prd[:,0])
        e_mbis = eval_dimer(RA, RB, ZA, ZB, QA,     QB,   )
        e_mpnn = eval_dimer(RA, RB, ZA, ZB, QA_prd, QB_prd)
        errs.append(e_mbis - e_mpnn)
        print(f'{df.sapt0_hfjdz_elst[i] * 627.509:8.3f} {e_mbis * 627.509:8.3f} {e_mpnn * 627.509:8.3f}')
        print(df.name[i].split('-'))
    errs = np.array(errs)
    print(np.average(np.abs(errs)) * 627.509)

    #df_elst = pd.DataFrame.from_dict(data=df_elst)
    #df_elst.to_pickle(f'preds/HBC6_elst.pkl', protocol=4)
    #print(df_elst)

    # older comment vvv


    #dfA = pd.read_pickle('data/S66x8-A.pkl')
    #dfA_prd = pd.read_pickle('preds/S66x8-A_modelB.pkl')
    #dfB = pd.read_pickle('data/S66x8-B.pkl')
    #dfB_prd = pd.read_pickle('preds/S66x8-B_modelB.pkl')

    #for i in range(66*8):

    #    if i % 8 == 0:
    #        print()
    #        print(dfA.name[i])
    #    RA = dfA.R[i]
    #    RB = dfB.R[i]
    #    ZA = dfA.Z[i]
    #    ZB = dfB.Z[i]
    #    QA = dfA.cartesian_multipoles[i]
    #    QB = dfB.cartesian_multipoles[i]
    #    QA_prd = dfA_prd.cartesian_multipoles_prd[i]
    #    QB_prd = dfB_prd.cartesian_multipoles_prd[i]
    #    #print(QA[:,0]-QA_prd[:,0])
    #    #print(QB[:,0]-QB_prd[:,0])
    #    _,_,e_mbis = eval_dimer(RA, RB, ZA, ZB, QA, QB, print_=False)
    #    _,_,e_mpnn = eval_dimer(RA, RB, ZA, ZB, QA_prd, QB_prd, print_=False)
    #    print(f'{e_mbis * 627.509:8.3f} {e_mpnn * 627.509:8.3f}')
