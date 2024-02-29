import numpy as np
from typing import Union
from CompChemUtils.chemdata import M



def center_of_mass(elems: Union[list, np.array], coords: np.array) -> np.array:
    ms = np.array([M[elem] for elem in elems])
    return np.dot(ms, coords) / np.sum(ms)
    


def moments_of_inertia(elems: Union[list, np.array], coords: np.array) -> (np.array, np.array):
    ms = np.array([M[elem] for elem in elems])
    totinertia = np.sum([m * np.dot(coord, coord) for m, coord in zip(ms, coords)])
    x, y, z = coords.T
    Ixx = np.sum(ms * (y**2 + z**2))
    Iyy = np.sum(ms * (x**2 + z**2))
    Izz = np.sum(ms * (x**2 + y**2))
    Ixy = -np.sum(ms * x * y)
    Iyz = -np.sum(ms * y * z)
    Ixz = -np.sum(ms * x * z)
    I = np.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]
    ])
    eigvals, eigvecs = np.linalg.eig(I)
    sortids = np.argsort(eigvals)[::-1]
    return totinertia, eigvals[sortids] / totinertia, eigvecs[:,sortids]



def right_handed(basis: np.array) -> bool:
    ref0 = np.cross(basis[:,1], basis[:,2])
    ref1 = np.cross(basis[:,2], basis[:,0])
    ref2 = np.cross(basis[:,0], basis[:,1])
    dot0 = np.dot(ref0, basis[:,0])
    dot1 = np.dot(ref1, basis[:,1])
    dot2 = np.dot(ref2, basis[:,2])
    return (dot0 > 0) and (dot1 > 0) and (dot2 > 0)



def pca(mat: np.array) -> np.array:
    mat = mat - np.mean(mat, axis=0)
    covmat = np.cov(mat.T)
    eigvals, eigvecs = np.linalg.eig(covmat)
    sortids = np.argsort(eigvals)[::-1]
    return eigvals[sortids], eigvecs[:,sortids]
