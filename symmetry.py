import numpy as np
from typing import Union
from CompChemUtils.transform import Householder


def mirror_plane(elems: Union[list, np.array], coords: np.array, n: np.array, tol: float=1e-2) -> int:
    M = Householder.mat(n)
    for i, coord1 in enumerate(coords):
        mirror_coord = np.dot(M, coord1)
        bestval = 1.0
        for j, coord2 in enumerate(coords):
            if elems[i] != elems[j]:
                continue
            diff = np.linalg.norm(mirrorcoords - coord1)
            if diff < bestval:
                bestval = diff
        if bestval > tol:
            return 0
    return 1