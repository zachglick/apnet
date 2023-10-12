from __future__ import annotations

import numpy as np
import qcelemental as qcel

from .pair_model import PairModel

"""
These functions are included for backwards compatibility with an old version of AP-Net
"""

def predict_sapt(dimers : list[qcel.models.Molecule], return_pairs=False, batch_size=200):

    if not isinstance(dimers, list):
        dimers = [dimers]

    # an ensemble of five models
    models = [PairModel.pretrained(ind) for ind in range(5)]
    
    if return_pairs:

        # predict SAPT IEs from each of the five ensemble members
        ies_pair = [model.predict_pairs(dimers) for model in models]

        ies_avg, ies_std = [], []
        for dimer_ind in range(len(dimers)):
            ies_pair_dimer = [ies_pair[model_ind][dimer_ind] for model_ind in range(5)]
            ies_pair_dimer = np.array(ies_pair_dimer)
            ies_pair_dimer = np.concatenate([np.sum(ies_pair_dimer, axis=1, keepdims=True), ies_pair_dimer], axis=1)
            ies_avg.append(np.average(ies_pair_dimer, axis=0))
            ies_std.append(np.std(ies_pair_dimer, axis=0))

        return ies_avg, ies_std

    else:

        # predict SAPT IEs from each of the five ensemble members
        ies = np.array([model.predict(dimers) for model in models])
        ies = np.concatenate([np.sum(ies, axis=2).reshape(5, -1, 1), ies], axis=2)

        ies_avg = np.average(ies, axis=0)
        ies_std = np.std(ies, axis=0)

        return ies_avg, ies_std

def predict_sapt_common(common_monomer, monomers, return_pairs=False, batch_size=200):

    if not isinstance(monomers, list):
        monomers = [monomers]

    common_monomer_str = common_monomer.to_string(dtype="xyz", units="angstrom").split("\n")[2:] 
    common_monomer_str = "\n".join(common_monomer_str).strip()
    common_monomer_charge = int(np.round(common_monomer.molecular_charge))
    common_monomer_str = f"{common_monomer_charge:d} 1\n" + common_monomer_str

    dimers = []

    for monomer in monomers:

        monomer_str = monomer.to_string(dtype="xyz", units="angstrom").split("\n")[2:] 
        monomer_str = "\n".join(monomer_str).strip()
        monomer_charge = int(np.round(monomer.molecular_charge))
        monomer_str = f"{monomer_charge:d} 1\n" + monomer_str

        dimer_str = common_monomer_str + "\n--\n" + monomer_str
        dimers.append(qcel.models.Molecule.from_data(dimer_str))

    return predict_sapt(dimers, return_pairs=return_pairs, batch_size=batch_size)


