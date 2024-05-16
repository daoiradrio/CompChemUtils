import numpy as np
from queue import Queue
from CompChemUtils.structure import Structure
from scipy.optimize import linear_sum_assignment
from CompChemUtils.symmetry import mirror_plane
from CompChemUtils.properties import moments_of_inertia, center_of_mass
from CompChemUtils.chemdata import M
from CompChemUtils.transform import Rxyz
from CompChemUtils.files import read_xyz_file



# ADD HUNGARIAN RMSD AND RESOLVE EDGE CASES WHERE THE ALGORITHM GETS STUCK AT THE WRONG ASSIGNMENT DUE TO SYMMETRY
# (SOMETIMES EVEN FAILS BECAUSE NO ATOM CAN BE ASSGINED IN THE BEGINNING DUE TO SYMMETRY)
class RMSD:

    def __init__(self):
        pass

    
    def kabsch(self, coords1: np.array, coords2: np.array) -> tuple:
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


    def kabsch_rmsd(self, coords1: np.array, coords2: np.array) -> float:
        Rmat = self.kabsch(coords1, coords2)
        coords2 = np.dot(coords2, Rmat)
        return np.sqrt(np.mean(np.linalg.norm(coords1 - coords2, axis=1)**2))
    

    def hungarian_rmsd(self, elems1: list, coords1: np.array, elems2: list, coords2: np.array) -> float:
        natoms = len(coords1.shape[0])
        costmat = np.zeros((natoms, natoms))
        #Rkabsch = self.kabsch(coords1, coords2)
        #coords2 = coords2 @ Rkabsch
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


    def tight_rmsd(self, file1: str, file2: str):
        mol1 = Structure()
        mol2 = Structure()
        mol1.set_structure_from_xyz_file(file1, as_matrix=False)
        mol2.set_structure_from_xyz_file(file2, as_matrix=False)
        return self.graph_based_rmsd(mol1, mol2)


    def graph_based_rmsd(self, mol1: Structure, mol2: Structure) -> float:
        coords1, coords2 = self.match_coords(mol1, mol2)
        Rmat = self.kabsch(coords1, coords2)
        coords2 = np.dot(coords2, Rmat)
        return np.sqrt(np.mean(np.linalg.norm(coords1 - coords2, axis=1)**2))


    def __spheres(self, connectivity: dict, elems: list, firstatomi: int) -> list:
        memory = [firstatomi]
        atoms = Queue()
        for atomi in connectivity[firstatomi]:
            atoms.put(atomi)
            memory.append(atomi)
        spheres = []
        sphere_counter = len(connectivity[firstatomi])
        next_sphere_counter = 0
        spherei = 0
        while not atoms.empty():
            spheres.append([])
            while sphere_counter:
                atomi = atoms.get()
                spheres[spherei].append(elems[atomi])
                sphere_counter -= 1
                for neighbori in connectivity[atomi]:
                    if not neighbori in memory:
                        memory.append(neighbori)
                        atoms.put(neighbori)
                        next_sphere_counter += 1
            spheres[spherei] = sorted(spheres[spherei])
            spherei += 1
            sphere_counter = next_sphere_counter
            next_sphere_counter = 0
        return spheres
    

    def match_coords(self, mol1: Structure, mol2: Structure) -> tuple:
        pairs = {}
        assigned1 = [0] * mol1.natoms
        assigned2 = [0] * mol2.natoms
        matchedcoords1 = []
        matchedcoords2 = []
        spheres1 = {atomi: self.__spheres(mol1.bond_dict, mol1.elems, atomi) for atomi in range(mol1.natoms)}
        spheres2 = {atomi: self.__spheres(mol2.bond_dict, mol2.elems, atomi) for atomi in range(mol2.natoms)}
        for atomi in range(mol1.natoms):
            eqs = []
            for atomj in range(mol2.natoms):
                if mol1.elems[atomi] != mol2.elems[atomj]:
                    continue
                if spheres1[atomi] == spheres2[atomj]:
                    eqs.append(atomj)
            if len(eqs) > 1:
                pairs[atomi] = eqs
            elif len(eqs) == 1:
                matchedcoords1.append(mol1.coords[atomi])
                matchedcoords2.append(mol2.coords[eqs[0]])
                assigned1[atomi] = 1
                assigned2[eqs[0]] = 1
            else:
                print()
                print("***************** WARNING *******************")
                print("RMSD MODULE: Molecules seem to be different.")
                print("*********************************************")
                print()
                return mol1.coords, mol2.coords
        matchedcoords1 = np.array(matchedcoords1)
        matchedcoords2 = np.array(matchedcoords2)
        if matchedcoords1.shape[0] == mol1.natoms and matchedcoords2.shape[0] == mol2.natoms:
            return matchedcoords1, matchedcoords2
        for curratomi, eqatoms in pairs.items():
            if assigned1[curratomi]:
                continue
            dref = np.zeros((matchedcoords1.shape[0], len(eqatoms)))
            dref.T[:,:] = np.linalg.norm(matchedcoords1 - mol1.coords[curratomi], axis=1)
            d = np.zeros((matchedcoords2.shape[0], len(eqatoms)))
            for col, eqatomi in enumerate(eqatoms):
                if assigned2[eqatomi]:
                    continue
                d[:,col] = np.linalg.norm(matchedcoords2 - mol2.coords[eqatomi], axis=1)
            minatomi = np.argmin(np.linalg.norm(dref - d, axis=0))
            minatomi = eqatoms[minatomi]
            assigned1[curratomi] = 1
            assigned2[minatomi] = 1
            matchedcoords1 = np.vstack((matchedcoords1, mol1.coords[curratomi]))
            matchedcoords2 = np.vstack((matchedcoords2, mol2.coords[minatomi]))
        return matchedcoords1, matchedcoords2


# TESTING NEW APPROACH BASED ON ALIGNING PRINCIPAL AXES OF INERTIA BEFORE APPLYING HUNGARIAN RMSD:


    def __check_mirror_planes(self, elems: list, coords: np.array, basis: np.array) -> np.array:
        planes = [0,0,0]
        for i in range(3):
            ax = basis[:,i]
            planes[i] = mirror_plane(elems, coords, ax)
        return np.array(planes)


    def __sym_unique_moi_basis(self, elems: list, coords: np.array) -> np.array:
        center = center_of_mass(elems, coords)
        coords -= center
        _, _, moibasis = moments_of_inertia(elems, coords)
        
        planes = self.__check_mirror_planes(elems, coords, moibasis)
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


    def __align_structures(self, basis1: np.array, coords1: np.array, basis2: np.array, coords2: np.array) -> tuple:
        for i in range(2):
            refvec = basis1[:,i]
            vec = basis2[:,i]
            Rmat = Rxyz.align_vec_mat(refvec, vec)
            basis2 = np.dot(Rmat, basis2)
            coords2 = np.dot(Rmat, coords2.T).T
        return basis2, coords2
    

    def moi_based_rmsd(self, file1: str, file2: str) -> float:
        # read in XYZ information
        _, elems1, coords1 = read_xyz_file(file1)
        _, elems2, coords2 = read_xyz_file(file2)
        # compute reference basis
        ref_basis = self.__sym_unique_moi_basis(elems1, coords2)
        # compute other basis
        basis = self.__sym_unique_moi_basis(elems2, coords2)
        # align both molecular structures by aligning reference basis and other basis
        _, coords2 = self.__align_structures(ref_basis, coords1, basis, coords2)
        # apply Hungarian-based RMSD
        return self.hungarian_rmsd(elems1, coords1, elems2, coords2)
