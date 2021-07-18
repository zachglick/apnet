"""
General utility functions for pre-processing molecules
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import qcelemental as qcel

import tensorflow as tf

import logging
tf.get_logger().setLevel(logging.ERROR)

from apnet import constants

def qcel_to_dimerdata(dimer):
    """ proper qcel mol to ML-ready numpy arrays """

    # this better be a dimer (not a monomer, trimer, etc.)
    if  len(dimer.fragments) != 2:
        #raise AssertionError(f"A dimer must have exactly 2 molecular fragments, found {len(dimer.fragments)}")
        return None

    ZA = dimer.symbols[dimer.fragments[0]]
    ZB = dimer.symbols[dimer.fragments[1]]

    # only some elements allowed
    try:
        ZA = np.array([constants.elem_to_z[za] for za in ZA])
        ZB = np.array([constants.elem_to_z[zb] for zb in ZB])
    except:
        return None

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

    return (R, Z, aQ)

def dimerdata_to_qcel(RA, RB, ZA, ZB, aQA, aQB):
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

def monomerdata_to_qcel(R, Z, aQ):
    """ ML-ready numpy arrays to qcel mol """

    n = R.shape[0]

    tQ = int(round(aQ * n))

    assert abs(tQ - aQ * n) < 1e-6

    block = f"{tQ} {1}\n"
    for i in range(n):
        block += f"{constants.z_to_elem[Z[i]]} {R[i,0]} {R[i,1]} {R[i,2]}\n"

    monomer = block + "no_com\nno_reorient\nunits angstrom"
    monomer = qcel.models.Molecule.from_data(monomer)
    return monomer

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

def load_dimer_dataset(file, max_size=None):
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

    if max_size is not None and max_size < N:
        df = df.head(max_size)
        N = max_size

    RA = df.RA.tolist()
    RB = df.RB.tolist()
    ZA = df.ZA.tolist()
    ZB = df.ZB.tolist()
    TQA = df.TQA.tolist()
    TQB = df.TQB.tolist()
    aQA = [TQA[i] / np.sum(ZA[i] > 0) for i in range(N)]
    aQB = [TQB[i] / np.sum(ZB[i] > 0) for i in range(N)]
    try:
        labels = df[['Total_aug', 'Elst_aug', 'Exch_aug', 'Ind_aug', 'Disp_aug']].to_numpy()
    except:
        labels = None

    dimers = []
    for i in range(N):
        dimers.append(dimerdata_to_qcel(RA[i], RB[i], ZA[i], ZB[i], aQA[i], aQB[i]))

    return dimers, labels


def load_monomer_dataset(file, max_size=None):
    """Load multiple monomers from a :class:`~pandas.DataFrame`

    Loads monomers from the :class:`~pandas.DataFrame` format associated with the original AP-Net publication.
    Each row of the :class:`~pandas.DataFrame` corresponds to a molecular dimer.

    The columns [`Z`, `R`, and `total_charge`] are required.
    `Z` is atom types (:class:`~numpy.ndarray` of `int` with shape (`n`,)).
    `R` is atomic positions in Angstrom (:class:`~numpy.ndarray` of `float` with shape (`n`,3)).
    `total_charge` are monomer charges (int).

    The columns [`cartesian_multipoles`, `volume_ratios`, and `valence_widths`] are optional.
    `cartesian_multipoles` describes atom-centered charges, dipoles, and quadrupoles (:class:`~numpy.ndarray` of `float` with shape (`n`, 10). The ordering convention is [q, u_x, u_y, u_z, Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz], all in a.u.)
    `volume_ratios` is the ratio of the volume of the atom-in-molecule to the free atom (:class:`~numpy.ndarray` of `float` with shape (`n`, 1), unitless
    `valence_widths` is the width describing the valence electron density (:class:`~numpy.ndarray` of `float` with shape (`n`, 1), TODO: check units. a.u. ? inverse width?

    Parameters
    ----------
    file : str
        The name of a file containing the :class:`~pandas.DataFrame`
    
    Returns
    -------
    monomers : list of :class:`~qcelemental.models.Molecule`
    cartesian_multipoles : list of :class:`~numpy.ndarray` or None
        None is returned if the `cartesian_multipoles` column is not present in the :class:`~pandas.DataFrame`
    volume_ratios : list of :class:`~numpy.ndarray` or None
        None is returned if the `volume_ratios` column is not present in the :class:`~pandas.DataFrame`
    valence_widths : list of :class:`~numpy.ndarray` or None
        None is returned if the `valence_widths` column is not present in the :class:`~pandas.DataFrame`
    """

    df = pd.read_pickle(file)
    N = len(df.index)

    if max_size is not None and max_size < N:
        df = df.head(max_size)
        N = max_size

    R = df.R.tolist()
    Z = df.Z.tolist()
    TQ = df.total_charge.tolist()
    aQ = [TQ[i] / np.sum(Z[i] > 0) for i in range(N)]

    try:
        cartesian_multipoles = df['cartesian_multipoles'].to_numpy()
    except:
        cartesian_multipoles = None

    try:
        volume_ratios = df['volume_ratios'].to_numpy()
    except:
        volume_ratios = None

    try:
        valence_widths = df['valence_widths'].to_numpy()
    except:
        valence_widths = None

    monomers = []
    for i in range(N):
        monomers.append(monomerdata_to_qcel(R[i], Z[i], aQ[i]))

    return monomers, cartesian_multipoles, volume_ratios, valence_widths


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

    mol2 = dimerdata_to_qcel(*data)
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


    #dimers, labels = load_dimer_dataset("data/200_dimers.pkl")
    #load_dimer_dataset("/theoryfs2/common/data/dimer-pickles/1600K_val_dimers-fixed.pkl")
    #print(labels)
