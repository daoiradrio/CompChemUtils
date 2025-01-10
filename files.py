import os
import numpy as np

from CompChemUtils.chemdata import nuclear_charges



def read_xyz_file(filepath: str, element_symbols=True):
    if not os.path.isfile(filepath):
        print()
        print("*********** ERROR ***********")
        print("File not found at given path.")
        print("*****************************")
        print()
        return None
    elems = list(np.loadtxt(filepath, skiprows=2, usecols=0, dtype=str))
    coords = np.loadtxt(filepath, skiprows=2, usecols=(1,2,3))
    num_atoms = len(elems)
    if not element_symbols:
          elems = np.array([nuclear_charges[elem] for elem in elems])
    return num_atoms, elems, coords


def write_xyz_file(filepath: str, elems: list, coords: np.array):
	with open(filepath, "w") as xyzfile:
		print(coords.shape[0], file=xyzfile, end="\n\n")
		for elem, (x,y,z) in zip(elems, coords):
			print(f"{elem}\t{x:18.10f}\t{y:18.10f}\t{z:18.10f}", file=xyzfile)


def write_file(filepath: str, contentstr: str):
    with open(filepath, "w") as outfile:
        print(contentstr, file=outfile)
    