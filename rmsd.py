import numpy as np

from queue import Queue
from structure import Structure

from visual import printmat, printvec



class RMSD():

    def __init__(self):
        pass


    def tight_rmsd(self, mol1: Structure, mol2: Structure) -> float:
        self.match_coords(mol1, mol2)


    # ADJUST SO THAT IT RETURN ROTATION MATRIX AND THEN MAKE IT ACCESSIBLE FOR USER
    def __kabsch(self, coords1: np.array, coords2: np.array) -> tuple:
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
        for i, _ in enumerate(coords2):
            coords2[i] = np.matmul(coords2[i], R)
        return coords1, coords2


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
            else:
                print("*** WARNING ***")
                print(print("RMSD MODULE: Molecules seems to be different."))
                return mol1.coords, mol2.coords
        matchedcoords1 = np.array(matchedcoords1)
        matchedcoords2 = np.array(matchedcoords2)
        #if matchedcoords1.shape[0] == mol1.natoms and matchedcoords2.shape[0] == mol2.natoms:
        #    return matchedcoords1, matchedcoords2
        for curratomi, eq_atoms in pairs.items():
            dref = np.zeros((matchedcoords1.shape[0], len(eq_atoms)))
            dref.T[:,:] = np.linalg.norm(matchedcoords1 - mol1.coords[curratomi], axis=1)
            #d = np.zeros((matchedcoords2.shape[0], len(eq_atoms)))