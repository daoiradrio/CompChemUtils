import os
import numpy as np



class Structure:

    def __init__(self, filepath: str=None):
        self.number_of_atoms = 0
        self.elems = None
        self.coords = None
        self.bond_partners = None
        self.bonds = None
        if filepath:
            self.compute_structure(filepath)
    
    def __compute_structure(self, filepath: str):
        if not os.path.exists(filepath):
            print("STRUCTURE MODULE: File not found at given path.")
            return
        self.read_xyz(filepath)
        self.compute_connectivity()
    
    def __read_xyz(self, filepath: str):
        self.elems = np.loadtxt(filepath, skiprows=2, usecols=0, dtype=str)
        self.coords = np.loadtxt(filepath, skiprows=2, usecols=(1,2,3))
    
    def __compute_connectivity(self):
        pass
