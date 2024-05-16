import numpy as np

from CompChemUtils.visual import printmat, printvec
from CompChemUtils.symmetry import mirror_plane
from CompChemUtils.properties import moments_of_inertia, center_of_mass
from CompChemUtils.files import read_xyz_file, write_xyz_file
from CompChemUtils.chemdata import M
from CompChemUtils.transform import Rxyz
from scipy.optimize import linear_sum_assignment
from NeuralNetworks.debug_funcs import get_plot_data



def check_mirror_planes(elems: list, coords: np.array, basis: np.array) -> np.array:
	planes = [0,0,0]
	for i in range(3):
		ax = basis[:,i]
		planes[i] = mirror_plane(elems, coords, ax)
	return np.array(planes)



def sym_unique_moi_basis(elems: list, coords: np.array):
	_, _, moibasis = moments_of_inertia(elems, coords)
	
	planes = check_mirror_planes(elems, coords, moibasis)
	nzero = planes.size - np.count_nonzero(planes)
	transformed_coords = np.dot(coords, moibasis)

	if nzero == 1:
		idx = np.where(planes == 0)[0][0]
		msumpos = 0.0
		msumneg = 0.0
		for i, coord in enumerate(transformed_coords):
			if coord[idx] < 0:
				msumneg += M[elems[i]]
			else:
				msumpos += M[elems[i]]
		if msumpos < msumneg:
			moibasis[:,idx] *= -1
	elif nzero == 2:
		idx = np.where(planes == 0)[0]
		for i in idx:
			msumpos = 0
			msumneg = 0
			for j, coord in enumerate(transformed_coords):
				if coord[i] < 0:
					msumneg += M[elems[j]]
				else:
					msumpos += M[elems[j]]
			if msumpos < msumneg:
				moibasis[:,i] *= -1
	elif nzero == 3:
		idx = np.where(planes == 0)[0]
		for i in idx[:-1]:
			msumpos = 0
			msumneg = 0
			for j, coord in enumerate(transformed_coords):
				if coord[i] < 0:
					msumneg += M[elems[j]]
				else:
					msumpos += M[elems[j]]
			if msumpos < msumneg:
				moibasis[:,i] *= -1
	
	return moibasis



def align_structures(basis1: np.array, coords1: np.array, basis2: np.array, coords2: np.array):
	for i in range(2):
		refvec = basis1[:,i]
		vec = basis2[:,i]
		Rmat = Rxyz.align_vec_mat(refvec, vec)
		basis2 = np.dot(Rmat, basis2)
		coords2 = np.dot(Rmat, coords2.T).T
	return basis2, coords2



def kabsch(coords1: np.array, coords2: np.array) -> tuple:
    center1 = np.mean(coords1, axis=0)
    center2 = np.mean(coords2, axis=0)
    coords1 -= center1
    coords2 -= center2
    H = np.matmul(coords1.T, coords2)
    U, S, Vt = np.linalg.svd(H)
    det = np.linalg.det(np.matmul(Vt.T, U.T))
    if det >= 0:
        det = 1.0
    else:
        det = -1.0
    matrix = np.array([
        [1, 0,  0 ],
        [0, 1,  0 ],
        [0, 0, det]
    ])
    R = np.matmul(np.matmul(Vt.T, matrix), U.T)
    return R



def hungarian_rmsd(elems1: list, coords1: np.array, elems2: list, coords2: np.array) -> float:
        natoms = coords1.shape[0]
        costmat = np.zeros((natoms, natoms))
        Rkabsch = kabsch(coords1, coords2)
        coords2 = coords2 @ Rkabsch
        for i in range(natoms):
            for j in range(natoms):
                elemterm = 0.0
                if elems1[i] != elems2[j]:
                    elemterm = 1000.0
                diffvec = coords1[i] - coords2[j]
                costval = np.dot(diffvec, diffvec) + elemterm
                costmat[i][j] = costval
        row, col = linear_sum_assignment(costmat)
        coords1, coords2 = coords1[row], coords2[col]
        return np.sqrt(np.mean(np.linalg.norm(coords1 - coords2, axis=1)**2))



# Read in XYZ information
#base_path = "/Users/dario/CompChemUtils/tests/testcases"
base_path = "/Users/dario/xyz_files"
file1 = f"{base_path}/L-Alanine.xyz"
file2 = f"{base_path}/L-Alanine.xyz"
natoms1, elems1, coords1 = read_xyz_file(file1)
natoms2, elems2, coords2 = read_xyz_file(file2)

Rmat1 = Rxyz.rand_rot_mat()
Rmat2 = Rxyz.rand_rot_mat()
coords1 = np.dot(coords1, Rmat1.T)
coords2 = np.dot(coords2, Rmat2.T)

# Compute reference basis
basis1 = sym_unique_moi_basis(elems1, coords1)

# Compute other basis
basis2 = sym_unique_moi_basis(elems2, coords2)

# Align both molecular structures by aligning reference basis and other basis
#basis2, coords2 = align_structures(basis1, coords1, basis2, coords2)

# Apply Hungarian-based RMSD
#rmsd = hungarian_rmsd(elems1, coords1, elems2, coords2)

#print(rmsd)

new_elems1, new_coords1 = get_plot_data(elems1, coords1, basis1)
new_elems2, new_coords2 = get_plot_data(elems2, coords2, basis2)
write_xyz_file("test1.xyz", new_elems1, new_coords1)
write_xyz_file("test2.xyz", new_elems2, new_coords2)

#coords1 = coords1 - center_of_mass(elems1, coords1)
#coords2 = coords2 - center_of_mass(elems2, coords2)
#write_xyz_file("test.xyz", elems1+elems2, np.concatenate((coords1, coords2)))

from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

#mol1 = Molecule.from_file(file1)
#mol2 = Molecule.from_file(file2)
mol1 = Molecule(elems1, np.dot(coords1, Rmat1.T))
mol2 = Molecule(elems2, np.dot(coords2, Rmat2.T))
analyzer1 = PointGroupAnalyzer(mol1)
analyzer2 = PointGroupAnalyzer(mol2)

#new_elems1, new_coords1 = get_plot_data(elems1, mol1.cart_coords, analyzer1.principal_axes)
#write_xyz_file("test1.xyz", new_elems1, new_coords1)
#new_elems2, new_coords2 = get_plot_data(elems2, mol2.cart_coords, analyzer2.principal_axes)
#write_xyz_file("test2.xyz", new_elems2, new_coords2)

'''
printvec(analyzer1.eigvals)
_, moi, _ = moments_of_inertia(elems1, coords1)
printvec(moi)
printvec(analyzer2.eigvals)
_, moi, _ = moments_of_inertia(elems2, coords2)
printvec(moi)
'''
