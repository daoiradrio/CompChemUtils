import os
import numpy as np
from chemdata import covalence_radii_single, covalence_radii_double, covalence_radii_triple



class Structure:

    def __init__(self, filepath: str=None):
        self.natoms = 0
        self.elems = None
        self.coords = None
        # STORE ONLY ONE HALF OF MATRIX, IMPLEMENT METHODS TO TRANSLATE BETWEEN HALF AND FULL MATRIX
        # IN GENERAL USE BETTER DATA STRUCTURE, A LOT OF ZEROS ARE STORED
        self.bonds = None
        if filepath:
            self.compute_structure(filepath)
    
    def compute_structure(self, filepath: str):
        if not os.path.exists(filepath):
            print("STRUCTURE MODULE: File not found at given path.")
            return
        self.read_xyz(filepath)
        self.__compute_connectivity()
    
    def read_xyz(self, filepath: str):
        self.elems = np.loadtxt(filepath, skiprows=2, usecols=0, dtype=str)
        self.coords = np.loadtxt(filepath, skiprows=2, usecols=(1,2,3))
        self.natoms = self.elems.size
    
    def __compute_connectivity(self):
        self.bonds = 10*np.identity(self.natoms)
        for i in range(self.natoms):
            for j in range(i+1, self.natoms):
                bond_order = self.__compute_bond_order(
                    self.elems[i],
                    self.coords[i],
                    self.elems[j],
                    self.coords[j]
                )
                if bond_order:
                    self.bonds[i][j] = bond_order
                    self.bonds[j][i] = bond_order
    
    def __compute_bond_order(self, elem1, coord1, elem2, coord2):
        tol = 0.08
        bond_order = 0
        d = np.linalg.norm(coord1 - coord2)
        if not elem1 == "H" and not elem2 == "H":
            double_bond = covalence_radii_double[elem1] + covalence_radii_double[elem2]
            triple_bond = covalence_radii_triple[elem1] + covalence_radii_triple[elem2]
            if d <= (triple_bond + tol):
                bond_order = 3
            elif d <= (double_bond + tol):
                bond_order = 2
        else:
            single_bond = covalence_radii_single[elem1] + covalence_radii_single[elem2]
            if d <= (single_bond + tol):
                bond_order = 1
        return bond_order
