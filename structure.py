import os
import numpy as np
from typing import Union
from CompChemUtils.chemdata import covalence_radii_single, covalence_radii_double, covalence_radii_triple



class Structure:

    def __init__(self, filepath: str=None):
        self.natoms = 0
        self.elems = None
        self.coords = None
        # FIND BETTER DATASTRUCTURE FOR CONNECTIVITY
        self.bonds = None
        if filepath:
            self.structure_from_xyzfile(filepath)
    

    def set_structure(self, elems: Union[list, np.array], coords: np.array) -> None:
        if not type(elems) == list:
            elems = list(elems)
        self.elems = elems
        self.coords = coords
        self.natoms = len(self.elems)
        self.__compute_connectivity()
    

    def set_structure_from_file(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            print()
            print("****************** WARNING ********************")
            print("STRUCTURE MODULE: File not found at given path.")
            print("***********************************************")
            return
        self.set_elems_coords_from_file(filepath)
        self.__compute_connectivity()
    

    def set_elems_coords(self, elems: list, coords: np.array) -> None:
        self.elems = elems
        self.coords = coords
        self.natoms = len(self.elems)


    def set_elems_coords_from_file(self, filepath: str) -> None:
        self.elems = list(np.loadtxt(filepath, skiprows=2, usecols=0, dtype=str))
        self.coords = np.loadtxt(filepath, skiprows=2, usecols=(1,2,3))
        self.natoms = len(self.elems)
    

    def __compute_connectivity(self) -> None:
        self.bonds = {i: [] for i in range(self.natoms)}
        for i in range(self.natoms):
            for j in range(i+1, self.natoms):
                bond_order = self.__compute_bond_order(
                    self.elems[i],
                    self.coords[i],
                    self.elems[j],
                    self.coords[j]
                )
                if bond_order:
                    self.bonds[i].append(j)
                    self.bonds[j].append(i)
    

    # EXTEND TO BOND ORDERS 2 AND 3
    def __compute_bond_order(self, elem1: str, coord1: np.array, elem2: str, coord2: np.array) -> int:
        tol = 0.08
        bond_order = 0
        d = np.linalg.norm(coord1 - coord2)
        single_bond = covalence_radii_single[elem1] + covalence_radii_single[elem2]
        if d <= (single_bond + tol):
                bond_order = 1
        # CRASHES IF ELEMENTS DO NOT HAVE COVALENCE RADII FOR DOUBLE AND TRIPLE BONDS
        #if not elem1 == "H" and not elem2 == "H":
        #    double_bond = covalence_radii_double[elem1] + covalence_radii_double[elem2]
        #    triple_bond = covalence_radii_triple[elem1] + covalence_radii_triple[elem2]
        #    if d <= (triple_bond + tol):
        #        bond_order = 3
        #    elif d <= (double_bond + tol):
        #        bond_order = 2
        return bond_order
    

    def write_xyz_file(self, filename) -> None:
        with open(filename, "w") as xyzfile:
            print(self.natoms, file=xyzfile, end="\n\n")
            for elem, (x,y,z) in zip(self.elems, self.coords):
                print(f"{elem}\t{x:18.10f}\t{y:18.10f}\t{z:18.10f}", file=xyzfile)
