"""
Fixed constants (units, atom types, etc.)
"""

import qcelemental as qcel

au2ang = qcel.constants.conversion_factor("bohr", "angstrom")

elem_to_z = { 
    'H'  : 1,
    'B'  : 3,
    'C'  : 6,
    'N'  : 7,
    'O'  : 8,
    'F'  : 9,
    'NA' : 11,
    'Na' : 11,
    'P'  : 15, 
    'S'  : 16, 
    'CL' : 17, 
    'Cl' : 17, 
    'BR' : 35, 
    'Br' : 35, 
}

z_to_elem = { 
    1  : 'H',
    3  : 'B',
    6  : 'C',
    7  : 'N',
    8  : 'O',
    9  : 'F',
    11 : 'Na',
    15 : 'P', 
    16 : 'S', 
    17 : 'Cl', 
    35 : 'Br', 
}

z_to_ind = {
    1  : 0,
    6  : 1,
    7  : 2,
    8  : 3,
    9  : 4,
    11 : 5,
    15 : 6,
    16 : 7,
    17 : 8,
    35 : 9,
}
