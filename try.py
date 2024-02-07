import os
import numpy as np
from CompChemUtils.structure import Structure
from CompChemUtils.rmsd import RMSD
from CompChemUtils.visual import printmat, printvec
from scipy.spatial.transform import Rotation as R



def rotate(coords, xangle=None, yangle=None, zangle=None):
    if xangle:
            rx = R.from_rotvec(xangle * np.array([1,0,0]))
            coords = np.dot(rx.as_matrix(), coords.T).T
    if yangle:
            ry = R.from_rotvec(yangle * np.array([0,1,0]))
            coords = np.dot(ry.as_matrix(), coords.T).T
    if zangle:
            rz = R.from_rotvec(zangle * np.array([0,0,1]))
            coords = np.dot(rz.as_matrix(), coords.T).T
    return coords



#file = f"/Users/dario/xyz_files/{os.sys.argv[1]}.xyz"
file1 = "/Users/dario/xyz_files/L-Alanine.xyz"
file2 = "/Users/dario/xyz_files/L-Alanine_different_order.xyz"
#file1 = "/Users/dario/xyz_files/H2O.xyz"
#file2 = "/Users/dario/xyz_files/H2O_different_order_rotated.xyz"
#cx, cy, cz = np.random.choice(np.arange(-1, 1.0001, 0.0001), 3)
mol1 = Structure(file1)
mol2 = Structure(file2)
#mol2.coords = rotate(mol2.coords, xangle=cx*np.pi, yangle=cy*np.pi, zangle=cz*np.pi)
#mol2.write_xyz_file("/Users/dario/xyz_files/H2O_different_order_rotated.xyz")
rmsdobj = RMSD()
rmsd = rmsdobj.tight_rmsd(mol1, mol2)
#coords1, coords2 = rmsdobj.match_coords(mol1, mol2)
print(rmsd)