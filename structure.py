import os
import numpy as np
from typing import Union
from CompChemUtils.chemdata import covalence_radii_single, covalence_radii_double, covalence_radii_triple
from CompChemUtils.files import read_xyz_file



class Structure:

    def __init__(self, filepath: str=None):
        self.natoms = 0
        self.elems = None
        self.coords = None
        self.bond_dict = None
        self.bond_mat = None # STORE ONLY UPPER OR LOWER TRIANGLE AND IMPLEMENT HASH FUNCTION
        if filepath:
            self.set_structure_from_xyz_file(filepath)
    

    def set_elems_and_coords(self, elems: Union[list, np.array], coords: np.array) -> None:
        if not type(elems) == list:
            elems = list(elems)
        self.elems = elems
        self.coords = coords
        self.natoms = len(elems)
    

    def set_elems_and_coords_from_xyz_file(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            print()
            print("****************** WARNING ********************")
            print("STRUCTURE MODULE: File not found at given path.")
            print("***********************************************")
            print()
            return
        self.natoms, self.elems, self.coords = read_xyz_file(filepath)
    

    def set_graph(self, as_matrix: bool=True) -> None:
        if self.natoms == 0:
            print()
            print("********************** WARNING ************************")
            print("STRUCTURE MODULE: ELEMENTS AND COORDINATES NOT SET YET.")
            print("*******************************************************")
            print()
            return
        if as_matrix:
            if self.bond_mat is not None:
                self.mat = None
            self.__compute_graph_as_matrix()
        else:
            if self.bond_dict is not None:
                self.bond_dict = None
            self.__compute_graph_as_dict()


    def set_structure(self, elems: Union[list, np.array], coords: np.array, as_matrix: bool=True) -> None:
        if not type(elems) == list:
            elems = list(elems)
        self.elems = elems
        self.coords = coords
        self.natoms = len(self.elems)
        if as_matrix:
            self.__compute_graph_as_matrix()
        else:
            self.__compute_graph_as_dict()
    

    def set_structure_from_xyz_file(self, filepath: str, as_matrix: bool=True) -> None:
        if not os.path.exists(filepath):
            print()
            print("****************** WARNING ********************")
            print("STRUCTURE MODULE: File not found at given path.")
            print("***********************************************")
            print()
            return
        self.natoms, self.elems, self.coords = read_xyz_file(filepath)
        if as_matrix:
            self.__compute_graph_as_matrix()
        else:
            self.__compute_graph_as_dict()
    

    def __compute_graph_as_matrix(self) -> None:
        self.bond_mat = np.zeros((self.natoms, self.natoms))
        for i in range(self.natoms):
            self.bond_mat[i][i] = i
            for j in range(i+1, self.natoms):
                bond_order = self.__compute_bond_order(
                    self.elems[i],
                    self.coords[i],
                    self.elems[j],
                    self.coords[j]
                )
                self.bond_mat[i][j] = bond_order
                self.bond_mat[j][i] = bond_order
    

    def __compute_graph_as_dict(self) -> None:
        self.bond_dict = {i: [] for i in range(self.natoms)}
        for i in range(self.natoms):
            for j in range(i+1, self.natoms):
                bond_order = self.__compute_bond_order(
                    self.elems[i],
                    self.coords[i],
                    self.elems[j],
                    self.coords[j]
                )
                if bond_order:
                    self.bond_dict[i].append(j)
                    self.bond_dict[j].append(i)
    

    def __compute_bond_order(self, elem1: str, coord1: np.array, elem2: str, coord2: np.array) -> int:
        tol = 0.08
        bond_order = 0
        d = np.linalg.norm(coord1 - coord2)
        single_bond = covalence_radii_single[elem1] + covalence_radii_single[elem2]
        double_bond = covalence_radii_double[elem1] + covalence_radii_double[elem2]
        triple_bond = covalence_radii_triple[elem1] + covalence_radii_triple[elem2]
        if d <= triple_bond + tol:
            bond_order = 3
        elif d <= double_bond + tol:
            bond_order = 2
        elif d <= single_bond + tol:
            bond_order = 1
        return bond_order
    

    @staticmethod
    def get_molecular_graph(elems, coords):
        tol = 0.08
        bond_idx = []
        bond_dist = []
        for i, elem1 in enumerate(elems):
            coord1 = coords[i]
            for j, elem2 in enumerate(elems[:i]):
                coord2 = coords[j]
                d = np.linalg.norm(coord1 - coord2)
                d_bond = covalence_radii_single[elem1] + covalence_radii_single[elem2]
                if d <= d_bond + tol:
                    bond_idx.append([i,j])
                    bond_dist.append(d)
        return bond_idx, bond_dist
