import numpy as np
from queue import Queue
from CompChemUtils.structure import Structure



# ADD HUNGARIAN RMSD AND RESOLVE EDGE CASES WHERE THE ALGORITHM GETS STUCK AT THE WRONG ASSIGNMENT DUE TO SYMMETRY
# (SOMETIMES EVEN FAILS BECAUSE NO ATOM CAN BE ASSGINED IN THE BEGINNING DUE TO SYMMETRY)
class RMSD:

    def __init__(self):
        pass

    
    @staticmethod
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


    def rmsd(self, coords1: np.array, coords2: np.array) -> float:
        Rmat = self.kabsch(coords1, coords2)
        coords2 = np.dot(coords2, Rmat)
        return np.sqrt(np.mean(np.linalg.norm(coords1 - coords2, axis=1)**2))
    

    def hungarian_rmsd(self, elems1: list, coords1: np.array, elems2: list, coords2: np.array) -> float:
        natoms = len(coords1.shape[0])
        costmat = np.zeros((n_atoms, n_atoms))
        Rkabsch = kabsch(coords1, coords2)
        coords2 = coords2 @ Rkabsch
        for i in range(natoms):
            for j in range(natoms):
                elemterm = 0.0
                if elems1[i] != elems2[j]:
                    elemterm = 1000.0
                diffvec = coords1[i] - coords2[j]
                costval = np.dot(diffvec, diffvec) + elemterm
                cost[i][j] = cost_value
        row, col = linear_sum_assignment(cost)
        coords1, coords2 = coords1[row], coords2[col]
        return np.sqrt(np.mean(np.linalg.norm(coords1 - coords2, axis=1)**2))


    def tight_rmsd(self, mol1: Structure, mol2: Structure) -> float:
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
        spheres1 = {atomi: self.__spheres(mol1.bonds, mol1.elems, atomi) for atomi in range(mol1.natoms)}
        spheres2 = {atomi: self.__spheres(mol2.bonds, mol2.elems, atomi) for atomi in range(mol2.natoms)}
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
