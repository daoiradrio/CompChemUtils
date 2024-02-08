import os
import numpy as np
from typing import Union
from CompChemUtils.chemdata import covalence_radii_single, covalence_radii_double, covalence_radii_triple
from CompChemUtils.files import read_xyz_file, write_xyz_file



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
            print()
            return
        self.set_elems_coords_from_file(filepath)
        self.__compute_connectivity()
    

    def set_elems_coords(self, elems: list, coords: np.array) -> None:
        self.elems = elems
        self.coords = coords
        self.natoms = len(self.elems)


    def set_elems_coords_from_file(self, filepath: str) -> None:
        self.natoms, self.elems, self.coords = read_xyz_file(filepath)
    

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
    
    
    def write_structure_to_XYZ(self, filepath: str) -> None:
        write_xyz_file(filepath, self.natoms, self.elems, self.coords)
