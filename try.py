import os
import numpy as np
from structure import Structure
from rmsd import RMSD
from visual import printmat, printvec



file = f"/Users/dario/xyz_files/{os.sys.argv[1]}.xyz"
mol1 = Structure(file)
mol2 = Structure(file)
rmsd = RMSD()
rmsd.tight_rmsd(mol1, mol2)