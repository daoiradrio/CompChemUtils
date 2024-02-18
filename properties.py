import numpy as np
from typing import Union
from CompChemUtils.chemdata import M



def moments_of_inertia(elems: Union[list, np.array], coords: np.array) -> (np.array, np.array):
    natoms = coords.shape[0]
    mx2y2 = 0.0
    mx2z2 = 0.0
    my2z2 = 0.0
    mxy = 0.0
    mxz = 0.0
    myz = 0.0
    for elem, (x,y,z) in zip(elems, coords):
        m = M[elem]
        mxy += m * x * y
        mxz += m * x * z
        myz += m * y * z
        mx2y2 += m * (x**2 + y**2)
        mx2z2 += m * (x**2 + z**2)
        my2z2 += m * (y**2 + z**2)
    I = np.array([
        [my2z2, -mxy, -mxz],
        [-mxy, mx2z2, -myz],
        [-mxz, -myz, mx2y2],
    ])
    eigvals, eigvecs = np.linalg.eig(I)
    sortids = np.argsort(eigvals)[::-1]
    return eigvals[sortids], eigvecs[:,sortids]
