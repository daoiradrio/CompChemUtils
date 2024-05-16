import torch
import numpy as np
from scipy import linalg as la
from CompChemUtils.transform import Rxyz
from CompChemUtils.chemdata import M, NAO
from CompChemUtils.symmetry import mirror_plane, linear
from CompChemUtils.properties import moments_of_inertia, center_of_mass
from e3x.so3.irreps import *
from e3x.config import Config
from e3x.so3._normalization import normalization_constant
from jax import numpy as jnp
	


'''
def collate_fn_varsize(batch):
	pad_val = 666

    batch_x_dftb, batch_y_delta, batch_distances, batch_edges = zip(*batch)
    max_n_rows = max([x_dftb.shape[-4] for x_dftb in batch_x_dftb])
    max_n_cols = max([edges.shape[-1] for edges in batch_edges])
    #max_ndistances = max([distances.shape[0] for distances in batch_distances])

    pad_batch_x_dftb = []
    pad_batch_y_delta = []
    pad_batch_edges = []
    for x_dftb, y_delta, edges in zip(batch_x_dftb, batch_y_delta, batch_edges):
        row_pad_len = max_n_rows - x_dftb.shape[-4]
        col_pad_len = max_n_cols - edges.shape[-1]
        pad_batch_x_dftb.append(
            torch.nn.functional.pad(
                torch.from_numpy(x_dftb),
                (0, 0, 0, 0, 0, 0, 0, row_pad_len),
                value=pad_val
            ).numpy()
        )
        pad_batch_y_delta.append(
            torch.nn.functional.pad(
                torch.from_numpy(y_delta),
                (0, 0, 0, 0, 0, 0, 0, row_pad_len),
                value=pad_val
            ).numpy()
        )
        edges[edges == edges.shape[-2]] = pad_val
        edges = torch.nn.functional.pad(
                    torch.from_numpy(edges),
                    (0, col_pad_len, 0, row_pad_len),
                    value=pad_val
                ).numpy()
        pad_batch_edges.append(edges)
    pad_batch_x_dftb = np.stack(pad_batch_x_dftb)
    pad_batch_y_delta = np.stack(pad_batch_y_delta)
    pad_batch_edges = np.stack(pad_batch_edges)
    
    pad_batch_distances = []
    for distances in batch_distances:
        row_pad_len = max_n_rows - distances.shape[-3]
        col_pad_len = max_n_cols - distances.shape[-2]
        pad_batch_distances.append(
            torch.nn.functional.pad(
                torch.from_numpy(distances),
                (0, 0, 0, col_pad_len, 0, row_pad_len),
                value=pad_val
            ).numpy()
        )
    pad_batch_distances = np.stack(pad_batch_distances)
    
    coeff_mask = pad_batch_x_dftb != pad_val
    coords_mask = pad_batch_edges != pad_val

    return pad_batch_x_dftb, pad_batch_y_delta, coeff_mask, pad_batch_distances, pad_batch_edges, coords_mask
'''



def check_mirror_planes(elems: list, coords: np.array, basis: np.array) -> np.array:
	planes = [0,0,0]
	for i in range(3):
		ax = basis[:,i]
		planes[i] = mirror_plane(elems, coords, ax)
	return np.array(planes)



# RUECKGABE VON BASIS UND KOORDINATEN NUR FUER ENTWICKLUNG NOETIG (UM EIGENBASIS ZU PLOTTEN)
# AM ENDE NUR ROTATIONSMATRIX ZURUECKGEBEN
def align_basis(basis: np.array, coords: np.array, idx: int=0) -> np.array:
	vec = basis[:, idx]
	refvec = np.zeros(3)
	refvec[idx] = 1.0
	Rmat = Rxyz.align_vec_mat(refvec, vec)
	return np.dot(Rmat, basis), np.dot(Rmat, coords.T).T, Rmat



def unique_orient(elems: list, coords: np.array):
	coords = coords - center_of_mass(elems, coords)
	_, moi, moibasis = moments_of_inertia(elems, coords)
	if moi[0] * moi[1] * moi[2] < 1e-10:
		coords = coords - np.mean(coords, axis=0)
		dists = np.linalg.norm(coords, axis=1)
		ids = np.argsort(dists)[::-1][:2]
		v1, v2 = coords[ids]
		ax = (v1 - v2) / np.linalg.norm(v1 - v2)
		Rmat = Rxyz.align_vec_mat(np.array([1,0,0]), ax)
		coords = np.dot(Rmat, coords.T).T
		moibasis = np.dot(Rmat, moibasis)
		return coords, moibasis, [1,1,1], Rmat

	planes = check_mirror_planes(elems, coords, moibasis)
	nzero = planes.size - np.count_nonzero(planes)
	Rmat = np.identity(3)

	for i in range(3):
		for j in range(3):
			canvec = np.zeros(3)
			canvec[j] = 1
			if np.allclose(moibasis[:,i], canvec):
				Rrand = Rxyz.rand_rot_mat()
				moibasis = np.dot(Rrand, moibasis)
				coords = np.dot(Rrand, coords.T).T
				Rmat = Rrand

	if nzero == 0:
		moibasis, coords, Rmat1 = align_basis(moibasis, coords, 0)
		moibasis, coords, Rmat2 = align_basis(moibasis, coords, 1)
		Rmat = Rmat2 @ Rmat1 @ Rmat
	elif nzero == 1:
		transcoords = np.dot(coords, moibasis)
		idx = np.where(planes == 0)[0][0]
		msumpos = 0.0
		msumneg = 0.0
		for i, coord in enumerate(transcoords):
			if coord[idx] < 0:
				msumneg += M[elems[i]]
			else:
				msumpos += M[elems[i]]
		if msumpos < msumneg:
			moibasis[:,idx] *= -1
		moibasis, coords, Rmat1 = align_basis(moibasis, coords, idx)
		moibasis, coords, Rmat2 = align_basis(moibasis, coords, [2,0,1][idx])
		Rmat = Rmat2 @ Rmat1 @ Rmat
	elif nzero == 2:
		transcoords = np.dot(coords, moibasis)
		idx = np.where(planes == 0)[0]
		for i in idx:
			msumpos = 0
			msumneg = 0
			for j, coord in enumerate(transcoords):
				if coord[i] < 0:
					msumneg += M[elems[j]]
				else:
					msumpos += M[elems[j]]
			if msumpos < msumneg:
				moibasis[:,i] *= -1
			moibasis, coords, tempRmat = align_basis(moibasis, coords, i)
			Rmat = tempRmat @ Rmat
	elif nzero == 3:
		transcoords = np.dot(coords, moibasis)
		idx = np.where(planes == 0)[0]
		for i in idx[:-1]:
			msumpos = 0
			msumneg = 0
			for j, coord in enumerate(transcoords):
				if coord[i] < 0:
					msumneg += M[elems[j]]
				else:
					msumpos += M[elems[j]]
			if msumpos < msumneg:
				moibasis[:,i] *= -1
			moibasis, coords, tempRmat = align_basis(moibasis, coords, i)
			Rmat = tempRmat @ Rmat
	
	return coords, Rmat



def get_ao_indices(elems: list):
	ao_indices = {}
	aoi = 0
	for i, elem in enumerate(elems):
		if elem == "H":
			ao_indices[f"H{i}"] = {
				"s": (aoi, aoi+1)
			}
			aoi += 1
		else:
			ao_indices[f"{elem}{i}"] = {
				"s": (aoi, aoi+1),
				"p": (aoi+1, aoi+4)
			}
			aoi += 4
	return ao_indices



def map_to_unique_xyz(elems: list, coords: np.array) -> (list, np.array):
	newcoords, Rmat = unique_orient(elems, coords)

	ms = []
	for elem in elems:
		if elem == "H":
			ms.append([1])
		else:
			ms.append([1])
			ms.append([1])
			ms.append(Rmat)
	Mtot = la.block_diag(*ms)
	'''
	colidx = np.full(len(elems), -1)
	naos = 0
	for i, elem in enumerate(elems):
		colidx[i] = naos
		naos += NAO["SZ"][elem]
	
	atomidx = [i for i, _ in enumerate(elems)]
	newcoords, atomidx = zip(*sorted(zip(newcoords, atomidx), key=lambda t: np.linalg.norm(t[0])))
	newcoords = np.array(newcoords)
	atomidx = list(atomidx)
	elems = np.array(elems)
	elems = elems[atomidx]
	
	row = 0
	col = 0
	colidx = colidx[atomidx]
	Pmat = np.zeros((naos,naos))
	for i, elem in enumerate(elems):
		s = NAO["SZ"][elem]
		col = colidx[i]
		Pmat[row:row+s, col:col+s] = np.identity(s)
		row += s
	Mtot = Pmat @ Mtot
	'''

	return newcoords, Rmat, Mtot

#lucasvisscher
#l.visscher@vu.nl

# decompose tensor product in to irreducible of degree l3
# approach is rather general, but here the last dimension of tensors x, y is assumend to be number of MOs
def reduce_mo_tensor_product(x, y, l1, l2, l3):
    xy = np.einsum("ik,jk->ijk", x, y).sum(2)
    xy = np.expand_dims(xy, axis=2)
    return normalization_constant(Config.normalization, l3) * (clebsch_gordan_for_degrees(l1, l2, l3) * xy).sum(1).sum(0)


# "invert" decomposition of tensor product (of tensors of degree l1, l2) into irreducible of degree l3
def dereduce_mo_tensor_product(x, l1, l2, l3):
    return ((x * clebsch_gordan_for_degrees(l1, l2, l3)) / normalization_constant(Config.normalization, l3)).sum(2)


def mean_squared_error(y_pred, y):
    return jnp.mean(optax.l2_loss(y_pred, y))


def mean_absolute_error(pred, target):
    return jnp.mean(jnp.abs(pred - target))
