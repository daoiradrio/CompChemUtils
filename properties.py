import numpy as np
from typing import Union
from CompChemUtils.chemdata import M



def center_of_mass(elems: Union[list, np.array], coords: np.array) -> np.array:
    com = np.array([0,0,0])
    mtot = 0.0
    for elem, coord in zip(elems, coords):
        com = com + M[elem] * coord
        mtot += M[elem]
    return com / mtot
    


def moments_of_inertia(elems: Union[list, np.array], coords: np.array) -> (np.array, np.array):
    coords = coords - center_of_mass(elems, coords)
    natoms = coords.shape[0]
    mx2y2 = 0.0
    mx2z2 = 0.0
    my2z2 = 0.0
    mxy = 0.0
    mxz = 0.0
    myz = 0.0
    totinertia = 0.0
    for elem, coord in zip(elems, coords):
        m = M[elem]
        x, y, z = coord
        mxy += m * x * y
        mxz += m * x * z
        myz += m * y * z
        mx2y2 += m * (x**2 + y**2)
        mx2z2 += m * (x**2 + z**2)
        my2z2 += m * (y**2 + z**2)
        totinertia += m * np.dot(coord, coord)
    I = np.array([
        [my2z2, -mxy, -mxz],
        [-mxy, mx2z2, -myz],
        [-mxz, -myz, mx2y2],
    ])
    eigvals, eigvecs = np.linalg.eig(I)
    sortids = np.argsort(eigvals)[::-1]
    return totinertia, eigvals[sortids] / totinertia, eigvecs[:,sortids]
