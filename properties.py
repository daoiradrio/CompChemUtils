import numpy as np
from typing import Union
from CompChemUtils.chemdata import M



def center_of_mass(elems: Union[list, np.array], coords: np.array) -> np.array:
    ms = np.array([M[elem] for elem in elems])
    return np.einsum("i, ij -> j", ms, coords) / np.sum(ms)
    


def moments_of_inertia(elems: Union[list, np.array], coords: np.array) -> (np.array, np.array):
    coords -= center_of_mass(elems, coords)
    I = np.zeros((3,3))
    masses = [M[elem] for elem in elems]
    totinertia = 0
    for i, m in enumerate(masses):
        coord = coords[i,:]
        for j in range(3):
            I[j,j] += m * (coord[(j+1) % 3]**2 + coord[(j+2) % 3]**2)
        for j, k in [(0, 1), (1, 2), (0, 2)]:
            I[j,k] += m * coord[j] * coord[k]
            I[k,j] += m * coord[k] * coord[j]
        totinertia += m * np.dot(coord, coord)
    eigvals, eigvecs = np.linalg.eig(I)
    sortids = np.argsort(eigvals)[::-1]
    return totinertia, eigvals[sortids] / totinertia, eigvecs[:,sortids]
    '''
    ms = np.array([M[elem] for elem in elems])
    totinertia = np.sum([m * np.dot(coord, coord) for m, coord in zip(ms, coords)])
    x, y, z = coords.T
    x2y2 = x**2 + y**2
    x2z2 = x**2 + z**2
    y2z2 = y**2 + z**2
    Ixx = np.einsum("i, i ->", ms, x2y2)
    Iyy = np.einsum("i, i ->", ms, x2z2)
    Izz = np.einsum("i, i ->", ms, x2y2)
    Ixy = -np.einsum("i, i, i ->", ms, x, y)
    Ixz = -np.einsum("i, i, i ->", ms, x, z)
    Iyz = -np.einsum("i, i, i ->", ms, y, z)
    '''
    '''
    Ixx = np.sum(ms * (y**2 + z**2))
    Iyy = np.sum(ms * (x**2 + z**2))
    Izz = np.sum(ms * (x**2 + y**2))
    Ixy = -np.sum(ms * x * y)
    Iyz = -np.sum(ms * y * z)
    Ixz = -np.sum(ms * x * z)
    '''
    '''
    I = np.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]
    ])
    eigvals, eigvecs = np.linalg.eig(I)
    sortids = np.argsort(eigvals)[::-1]
    return totinertia, eigvals[sortids] / totinertia, eigvecs[:,sortids]
    '''



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
