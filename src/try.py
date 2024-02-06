import os
from structure import Structure



file = f"/Users/dario/xyz_files/{os.sys.argv[1]}.xyz"
mol = Structure(file)

print(mol.bonds)