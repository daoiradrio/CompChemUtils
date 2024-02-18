import numpy as np
from typing import Union
from CompChemUtils.transform import Householder
from CompChemUtils.properties import center_of_mass


def mirror_plane(elems: Union[list, np.array], coords: np.array, n: np.array, tol: float=0.5) -> int:
    coords = coords - center_of_mass(elems, coords)
    M = Householder.mat(n)
    for elem1, coord1 in zip(elems, coords):
        mirrorcoord = np.dot(M, coord1)
        bestval = 1000.0
        for elem2, coord2 in zip(elems, coords):
            if elem1 != elem2:
                continue
            diff = np.linalg.norm(mirrorcoord - coord1)
            if diff < bestval:
                bestval = diff
        if bestval > tol:
            return 0
    return 1