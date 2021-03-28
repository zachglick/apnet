"""
General utility functions for pre-processing molecules and features
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import qcelemental as qcel

from apnet import features
from apnet import constants

def int_to_onehot(arr):
    """ arrs is a numpy array of integers w/ dims [NATOM]"""
    assert len(arr.shape) == 1
    arr2 = np.zeros((arr.shape[0], len(constants.z_to_ind)), dtype=np.int)
    for i, z in enumerate(arr):
        if z > 0:
            arr2[i, constants.z_to_ind[z]] = 1
    return arr2

def inflate(GA, GB):
    """ GA is the ACSFs of all monomer A atoms with dimensions [NATOMA x NMU x NZ]
        GB is the ACSFs of all monomer B atoms with dimensions [NATOMB x NMU x NZ]
        This function tiles GA and GB so that the first index is a pair of atoms
        Returns GA_ and GB_ both with dimensions [(NATOMA * NATOMB) x NMU x NZ]
        10/13/2020: Updated to also inflate multipoles, dimensions [NATOMA x 10]
     """
    nA, nB = GA.shape[0], GB.shape[0]
    if len(GA.shape) == 3:
        GA_ = np.expand_dims(GA, 1)
        GA_ = np.tile(GA_, (1,nB,1,1))
        GA_ = GA_.reshape(GA_.shape[0] * GA_.shape[1], GA_.shape[2], GA_.shape[3])
        GB_ = np.expand_dims(GB, 1)
        GB_ = np.tile(GB_, (1,nA,1,1))
        GB_ = np.transpose(GB_, (1,0,2,3))
        GB_ = GB_.reshape(GB_.shape[0] * GB_.shape[1], GB_.shape[2], GB_.shape[3])
        return GA_, GB_
    elif len(GA.shape) == 2:
        GA_ = np.expand_dims(GA, 1)
        GA_ = np.tile(GA_, (1,nB,1))
        GA_ = GA_.reshape(GA_.shape[0] * GA_.shape[1], GA_.shape[2])
        GB_ = np.expand_dims(GB, 1)
        GB_ = np.tile(GB_, (1,nA,1))
        GB_ = np.transpose(GB_, (1,0,2))
        GB_ = GB_.reshape(GB_.shape[0] * GB_.shape[1], GB_.shape[2])
        return GA_, GB_

def get_dimers(dataset):
    """
    Get molecular dimer (atoms and coordinates) and SAPT0 labels for a specified dataset

    Args:
        dataset: string corresponding to name of dataset

    Returns tuple of 
    Each element of the tuple is a 
    """

    # load dimer data
    if not os.path.isfile(f'data/{dataset}.pkl'):
       raise Exception(f'No dataset found at data/{dataset}.pkl')
    df = pd.read_pickle(f'data/{dataset}.pkl')

    # extract atom types and atomic coordinates
    ZA = df['ZA'].tolist()
    ZB = df['ZB'].tolist()
    RA = df['RA'].tolist()
    RB = df['RB'].tolist()
    TQA = df['TQA'].tolist()
    TQB = df['TQB'].tolist()

    # number of atoms in the monomers
    nA = [np.sum(za > 0) for za in ZA]
    nB = [np.sum(zb > 0) for zb in ZB]

    # average atomic charge of each monomer
    aQA = [TQA[i] / nA[i] for i in range(len(nA))]
    aQB = [TQB[i] / nB[i] for i in range(len(nB))]

    dimer = list(zip(RA, RB, ZA, ZB, aQA, aQB))

    # extract interaction energy label (if specified for the datset)
    try:
        sapt = df[['Elst_aug', 'Exch_aug', 'Ind_aug', 'Disp_aug']].to_numpy()
    except:
        sapt = None

    return dimer, sapt
    #return dimer, None


def make_features(RA, RB, ZA, ZB, QA, QB, MTP, ACSF_nmu=43, APSF_nmu=21, ACSF_eta=100, APSF_eta=25, elst_cutoff=5.0):

    nA = RA.shape[0]
    nB = RB.shape[0]
                                                        
    GA, GB, IA, IB = features.calculate_dimer(RA, RB, ZA, ZB)
    GA, GB = inflate(GA, GB)
    QA, QB = inflate(QA, QB)

    # append 1/D and cutoff to D
    RAB = distance_matrix(RA, RB)
    mask = (RAB <= elst_cutoff).astype(np.float64)
    cutoff = 0.5 * (np.cos(np.pi * RAB / elst_cutoff) + 1) * mask
    RAB = np.stack([RAB, 1.0 / RAB, cutoff], axis=-1)

    # append onehot(Z) to Z
    ZA = np.concatenate([ZA.reshape(-1,1), int_to_onehot(ZA)], axis=1)
    ZB = np.concatenate([ZB.reshape(-1,1), int_to_onehot(ZB)], axis=1)

    # tile ZA by atoms in monomer B and vice versa
    ZA = np.expand_dims(ZA, axis=1)
    ZA = np.tile(ZA, (1, nB, 1))
    ZB = np.expand_dims(ZB, axis=0)
    ZB = np.tile(ZB, (nA,1,1))


    #ZA = ZA.astype(float)
    #ZB = ZA.astype(float)

    # flatten the NA, NB indices
    ZA = ZA.reshape((-1,) + ZA.shape[2:]) 
    ZB = ZB.reshape((-1,) + ZB.shape[2:]) 
    RAB = RAB.reshape((-1,) + RAB.shape[2:])
    IA = IA.reshape((-1,) + IA.shape[2:])
    IB = IB.reshape((-1,) + IB.shape[2:])
    MTP = np.expand_dims(MTP.reshape((-1,)), axis=-1)
 
    # APSF is already made per atom pair 
    # We won't tile ACSFs (which are atomic) into atom pairs b/c memory, do it at runtime instead

    # these are the final shapes:
    # ZA[i]  shape: NA * NB x (NZ + 1)
    # ZB[i]  shape: NA * NB x (NZ + 1)
    # GA[i]  shape: NA x NMU1 x NZ
    # GB[i]  shape: NB x NMU1 x NZ
    # IA[i]  shape: NA * NB x NMU2 x NZ
    # IB[i]  shape: NA * NB x NMU2 x NZ
    # RAB[i] shape: NA * NB x 3
    # y[i]   scalar

    return (ZA, ZB, RAB, GA, GB, IA, IB, QA, QB, MTP)

def qcel_to_dimerdata(dimer):
    """ proper qcel mol to ML-ready numpy arrays """

    # this better be a dimer
    if  len(dimer.fragments) != 2:
        raise AssertionError(f"A dimer must have exactly 2 molecular fragments, found {len(dimer.fragments)}")

    ZA = dimer.symbols[dimer.fragments[0]]
    ZB = dimer.symbols[dimer.fragments[1]]
    ZA = np.array([constants.elem_to_z[za] for za in ZA])
    ZB = np.array([constants.elem_to_z[zb] for zb in ZB])

    RA = dimer.geometry[dimer.fragments[0]] * constants.au2ang
    RB = dimer.geometry[dimer.fragments[1]] * constants.au2ang

    nA = len(dimer.fragments[0])
    nB = len(dimer.fragments[1])
    aQA = dimer.fragment_charges[0] / nA
    aQB = dimer.fragment_charges[1] / nB

    return (RA, RB, ZA, ZB, aQA, aQB)

def qcel_to_monomerdata(monomer):
    """ proper qcel mol to ML-ready numpy arrays """

    # this better be a monomer 
    if  len(monomer.fragments) != 1:
        raise AssertionError(f"A monomer must have exactly 1 molecular fragment, found {len(monomer.fragments)}")

    Z = monomer.symbols
    Z = np.array([constants.elem_to_z[z] for z in Z])

    R = monomer.geometry * constants.au2ang

    n = len(monomer.symbols)
    aQ = monomer.molecular_charge / n
    print(Z.shape, R.shape)

    return (R, Z, aQ)


def data_to_qcel(RA, RB, ZA, ZB, aQA, aQB):
    """ ML-ready numpy arrays to qcel mol """

    nA = RA.shape[0]
    nB = RB.shape[0]

    tQA = int(round(aQA * nA))
    tQB = int(round(aQB * nB))

    assert abs(tQA - aQA * nA) < 1e-6
    assert abs(tQB - aQB * nB) < 1e-6

    blockA = f"{tQA} {1}\n"
    for ia in range(nA):
        blockA += f"{constants.z_to_elem[ZA[ia]]} {RA[ia,0]} {RA[ia,1]} {RA[ia,2]}\n"

    blockB = f"{tQB} {1}\n"
    for ib in range(nB):
        blockB += f"{constants.z_to_elem[ZB[ib]]} {RB[ib,0]} {RB[ib,1]} {RB[ib,2]}\n"

    dimer = blockA + "--\n" + blockB + "no_com\nno_reorient\nunits angstrom"
    dimer = qcel.models.Molecule.from_data(dimer)
    return dimer

def load_bms_dimer(file):
    """Load a single dimer from the BMS-xyz format

    This function expects an xyz file in the format used with the 1.66M dimer dataset. 
    The first line contains the number of atoms. 
    The second line contains a comma-separated list of values such as the dimer name, dimer and monomer charges, SAPT labels (at various levels of theory), and number of atoms in the first monomer.
    The next `natom` lines each contain an atomic symbol follwed by the x, y, and z cooordinates of the atom (Angstrom)

    Parameters
    ----------
    file : str
        The name of a file containing the xyz
    
    Returns
    -------
    dimer : :class:`~qcelemental.models.Molecule`
    labels : :class:`~numpy.ndarray`
        The SAPT0/aug-cc-pV(D+d)Z interaction energy labels: [total, electrostatics, exchange, induction, and dispersion].
    """

    lines = open(file, 'r').readlines()

    natom = int(lines[0].strip())
    dimerinfo = (''.join(lines[1:-natom])).split(',')
    geom = lines[-natom:]


    nA = int(dimerinfo[-1])
    nB = natom - nA
    TQ = int(dimerinfo[1])
    TQA = int(dimerinfo[2])
    TQB = int(dimerinfo[3])
    assert TQ == (TQA + TQB)

    e_tot_aug = float(dimerinfo[14])
    e_elst_aug = float(dimerinfo[15])
    e_exch_aug = float(dimerinfo[16])
    e_ind_aug = float(dimerinfo[17])
    e_disp_aug = float(dimerinfo[18])

    assert abs(e_tot_aug  - (e_elst_aug + e_exch_aug + e_ind_aug + e_disp_aug)) < 1e-6

    blockA = f"{TQA} 1\n" + "".join(geom[:nA])
    blockB = f"{TQB} 1\n" + "".join(geom[nA:])
    dimer = blockA + "--\n" + blockB + "no_com\nno_reorient\nunits angstrom"
    dimer = qcel.models.Molecule.from_data(dimer)

    label = np.array([e_tot_aug, e_elst_aug, e_exch_aug, e_ind_aug, e_disp_aug])
    return dimer, label


def load_pickle(file):
    """Load multiple dimers from a :class:`~pandas.DataFrame`

    Loads dimers from the :class:`~pandas.DataFrame` format associated with the original AP-Net publication.
    Each row of the :class:`~pandas.DataFrame` corresponds to a molecular dimer.

    The columns [`ZA`, `ZB`, `RA`, `RB`, `TQA`, `TQB`] are required.
    `ZA` and `ZB` are atom types (:class:`~numpy.ndarray` of `int` with shape (`n`,)).
    `RA` and `RB` are atomic positions in Angstrom (:class:`~numpy.ndarray` of `float` with shape (`n`,3.)).
    `TQA` and `TQB` are monomer charges (int).

    The columns [`Total_aug`, `Elst_aug`, `Exch_aug`, `Ind_aug`, and `Disp_aug`] are optional.
    Each column describes SAPT0/aug-cc-pV(D+d)Z labels in kcal / mol (`float`).

    Parameters
    ----------
    file : str
        The name of a file containing the :class:`~pandas.DataFrame`
    
    Returns
    -------
    dimers : list of :class:`~qcelemental.models.Molecule`
    labels : list of :class:`~numpy.ndarray` or None
        None is returned if SAPT0 label columns are not present in the :class:`~pandas.DataFrame`
    """

    df = pd.read_pickle(file)
    N = len(df.index)

    RA = df.RA.tolist()
    RB = df.RB.tolist()
    ZA = df.ZA.tolist()
    ZB = df.ZB.tolist()
    TQA = df.TQA.tolist()
    TQB = df.TQB.tolist()
    aQA = [TQA[i] / np.sum(ZA[i] > 0) for i in range(N)]
    aQB = [TQB[i] / np.sum(ZB[i] > 0) for i in range(N)]
    labels = df[['Total_aug', 'Elst_aug', 'Exch_aug', 'Ind_aug', 'Disp_aug']].to_numpy()

    dimers = []
    for i in range(N):
        dimers.append(data_to_qcel(RA[i], RB[i], ZA[i], ZB[i], aQA[i], aQB[i]))

    return dimers, labels


if __name__ == "__main__":

    mol = qcel.models.Molecule.from_data("""
    0 1
    O 0.000000 0.000000 0.100000
    H 1.000000 0.000000 0.000000
    CL 0.000000 1.000000 0.400000
    --
    0 1
    O -4.100000 0.000000 0.000000
    H -3.100000 0.000000 0.200000
    O -4.100000 1.000000 0.100000
    H -4.100000 2.000000 0.100000
    no_com
    no_reorient
    units angstrom
    """)
    print(mol.to_string("psi4"))
    print(mol)

    data = qcel_to_dimerdata(mol)

    mol2 = data_to_qcel(*data)
    print(mol2.to_string("psi4"))
    print(mol2)


    mol3 = qcel.models.Molecule.from_data("""
    -2 1
    O -4.100000 0.000000 0.000000
    H -3.100000 0.000000 0.200000
    O -4.100000 1.000000 0.100000
    H -4.100000 2.000000 0.100000
    no_com
    no_reorient
    units angstrom
    """)

    R, Z, aQ = qcel_to_monomerdata(mol3)
    print(mol3)
    print(R)
    print(Z)
    print(aQ)


    #dimers, labels = load_pickle("data/200_dimers.pkl")
    #load_pickle("/theoryfs2/common/data/dimer-pickles/1600K_val_dimers-fixed.pkl")
    #print(labels)
