import warnings
import numpy as np
from math import factorial
from random import random
from scipy.spatial.transform import Rotation as R
from CompChemUtils.chemdata import NAO, basis_trans_mats
from CompChemUtils.visual import printmat



#TODO
# - class templates for rotations, currently lots repetitions among Rx, Ry and Rz (vielleicht sogar für sämtliche Transformationen zusammen?)
# - add block decomposition of DFTB Fock matrices into atomic (self) interaction blocks here?



class Householder:

    def __init__(self):
        pass
    
    @staticmethod
    def mat(n: np.array) -> np.array:
        n = n / np.linalg.norm(n)
        return np.identity(n.size) - 2*np.outer(n,n)



class H:

    def __init__(self):
        pass

    
    @staticmethod
    def orthogonalize_basis(H: np.array, S: np.array) -> np.array:
        nmos = S.shape[0]
        _, U = np.linalg.eig(S)
        U = np.matrix(U)
        s = np.matmul(U.H, np.matmul(S, U))
        for i in range(nmos):
            s[i,i] = 1.0 / np.sqrt(s[i,i])
        X = np.matmul(U, np.matmul(s, U.H))
        Hortho = np.matmul(X, np.matmul(H, X))
        return np.asarray(Hortho)
    

    @staticmethod
    def _reorder_rose(H: np.array, CS12: np.array, basis_set: str, elems: list, atoms_order: list) -> np.array:
        natoms = len(elems)
        nao1 = sum([NAO[basis_set][elem] for elem in elems])
        nmo2 = sum([NAO["SZ"][elem] for elem in elems])
        entry_length_row = np.zeros(natoms)
        entry_length_col = np.zeros(natoms)
        for i, j in enumerate(atoms_order):
            entry_length_row[i] = NAO["SZ"][elems[i]]
            entry_length_col[i] = NAO[basis_set][elems[j-1]]
        cumsum_row = np.cumsum(entry_length_row[:-1])
        row_start_index = np.concatenate((np.array([0]), cumsum_row))
        cumsum_col = np.cumsum(entry_length_col[:-1])
        col_start_index_temp = np.concatenate((np.array([0]), cumsum_col))
        col_start_index = [0] * natoms
        for i, j in enumerate(atoms_order):
            col_start_index[j - 1] = col_start_index_temp[i]

        M = np.zeros((nmo2, nao1))

        for i, elem in enumerate(elems):
            vals = basis_trans_mats[basis_set][elem]
            n, m = vals.shape
            j = int(row_start_index[i])
            k = int(col_start_index[i])
            M[j:j+n, k:k+m] = vals

        MCS12 = np.dot(M, CS12)

        IAO_order = [-1] * nmo2
        for i in range(nmo2):
            max_j = 0
            for j in range(1, nmo2):
                if abs(MCS12[i][max_j]) < abs(MCS12[i][j]):
                    max_j = j
            IAO_order[i] = max_j

        Hnew = np.zeros((nmo2,nmo2))

        for i in range(nmo2):
            for j in range(nmo2):
                Hnew[i,j] = H[IAO_order[i],IAO_order[j]]
        
        return Hnew



class Rxyz:

    def __init__(self, xangle: float=None, yangle: float=None, zangle: float=None):
        self.mat = None
        self.xangle = xangle
        self.yangle = yangle
        self.zangle = zangle
        if xangle and yangle and zangle:
            self.set_mat()
    

    def set_mat(self) -> None:
        self.mat = Rx.mat(self.xangle) @ Ry.mat(self.yangle) @ Rz.mat(self.zangle)
    

    @staticmethod
    def align_vec_mat(refvec: np.array, vec: np.array) -> np.array:
        angle = np.arccos(np.dot(refvec, vec))
        if angle > 1e-8:
            rotvec = np.cross(vec, refvec)
            rotvec = rotvec / np.linalg.norm(rotvec)
            r = R.from_rotvec(angle * rotvec)
            return r.as_matrix()
        else:
            return np.identity(3)
    

    @staticmethod
    def rand_rot(coords: np.array) -> np.array:
        cx, cy, cz = np.random.choice(np.arange(-1, 1.0001, 0.0001), 3)
        rotmat = Rx.mat(cx * np.pi) @ Ry.mat(cy * np.pi) @ Rz.mat(cz * np.pi)
        return np.dot(rotmat, coords)
    


class Rx:

    def __init__(self, angle: float=None):
        self.mat = None
        if angle != None:
            self.set_mat(angle)
    

    def set_mat(self, angle: float, degrees=False) -> None:
        if degrees:
            angle = np.deg2rad(angle)
        self.mat = R.from_rotvec(angle * np.array([1,0,0])).as_matrix()
    

    def get_mat(self) -> np.array:
        if self.mat == None:
            print()
            print("******************* WARNING *******************")
            print("X ROTATION MODULE: Matrix has not yet been set.")
            print("***********************************************")
            print()
        return self.mat
    

    @staticmethod
    def mat(angle: float, degrees=False) -> np.array:
        if degrees:
            angle = np.deg2rad(angle)
        return R.from_rotvec(angle * np.array([1,0,0])).as_matrix()
    

    @staticmethod
    def rotate(vec: np.array, angle: float, degrees=False) -> np.array:
        if degrees:
            angle = np.deg2rad(angle)
        rotmat = R.from_rotvec(angle * np.array([1,0,0])).as_matrix()
        return np.dot(rotmat, vec)



class Ry:

    def __init__(self, angle: float=None):
        self.mat = None
        if angle != None:
            self.set_mat(angle)
    

    def set_mat(self, angle: float, degrees=False) -> None:
        if degrees:
            angle = np.deg2rad(angle)
        self.mat = R.from_rotvec(angle * np.array([0,1,0])).as_matrix()
    

    def get_mat(self) -> np.array:
        if self.mat == None:
            print()
            print("******************* WARNING *******************")
            print("Y ROTATION MODULE: Matrix has not yet been set.")
            print("***********************************************")
            print()
        return self.mat
    

    @staticmethod
    def mat(angle: float, degrees=False) -> np.array:
        if degrees:
            angle = np.deg2rad(angle)
        return R.from_rotvec(angle * np.array([0,1,0])).as_matrix()
    

    @staticmethod
    def rotate(vec: np.array, angle: float, degrees=False) -> np.array:
        if degrees:
            angle = np.deg2rad(angle)
        rotmat = R.from_rotvec(angle * np.array([0,1,0])).as_matrix()
        return np.dot(rotmat, vec)



class Rz:

    def __init__(self, angle: float=None):
        self.mat = None
        if angle != None:
            self.set_mat(angle)
    

    def set_mat(self, angle: float, degrees=False) -> None:
        if degrees:
            angle = np.deg2rad(angle)
        self.mat = R.from_rotvec(angle * np.array([0,0,1])).as_matrix()
    

    def get_mat(self) -> np.array:
        if self.mat == None:
            print()
            print("******************* WARNING *******************")
            print("Z ROTATION MODULE: Matrix has not yet been set.")
            print("***********************************************")
            print()
        return self.mat
    

    @staticmethod
    def mat(angle: float, degrees=False) -> np.array:
        if degrees:
            angle = np.deg2rad(angle)
        return R.from_rotvec(angle * np.array([0,0,1])).as_matrix()
    

    @staticmethod
    def rotate(vec: np.array, angle: float, degrees=False) -> np.array:
        if degrees:
            angle = np.deg2rad(angle)
        rotmat = R.from_rotvec(angle * np.array([0,0,1])).as_matrix()
        return np.dot(rotmat, vec)



class Tesseral:

    def __init__(self, l: int=None):
        self.l = None
        self.mat = None
        if l != None:
            self.l = l
            self.set_mat(l)


    def set_mat(self, l: int) -> None:
        Tmat = np.zeros((2*l+1, 2*l+1), dtype=np.complex128)
        Tmat[l,l] = 1.0
        rfac = 1.0 / np.sqrt(2)
        ifac = 1j * rfac
        for i, m1 in enumerate(range(-l, l+1)):
            for j, m2 in enumerate(range(-l, l+1)):
                if np.abs(m1) != np.abs(m2):
                    continue
                if m1 < 0:
                    if m2 < 0:
                        Tmat[i][j] = ifac
                    elif m2 > 0:
                        Tmat[i][j] = (-1)**m1 * (-ifac)
                elif m1 > 0:
                    if m2 > 0:
                        Tmat[i][j] = (-1)**m1 * rfac
                    elif m2 < 0:
                        Tmat[i][j] = rfac
        self.mat = Tmat
    

    def get_mat(self) -> np.array:
        if self.mat == None:
            print()
            print("****************** WARNING ******************")
            print("TESSERAL MODULE: Matrix has not yet been set.")
            print("*********************************************")
            print()
        return self.mat
    

    @staticmethod
    def mat(l: int) -> np.array:
        Tmat = np.zeros((2*l+1, 2*l+1), dtype=np.complex128)
        Tmat[l,l] = 1.0
        rfac = 1.0 / np.sqrt(2)
        ifac = 1j * rfac
        for i, m1 in enumerate(range(-l, l+1)):
            for j, m2 in enumerate(range(-l, l+1)):
                if np.abs(m1) != np.abs(m2):
                    continue
                if m1 < 0:
                    if m2 < 0:
                        Tmat[i][j] = ifac
                    elif m2 > 0:
                        Tmat[i][j] = (-1)**m1 * (-ifac)
                elif m1 > 0:
                    if m2 > 0:
                        Tmat[i][j] = (-1)**m1 * rfac
                    elif m2 < 0:
                        Tmat[i][j] = rfac
        return Tmat



class WignerD:

    def __init__(self, Rmat: np.array=None, l: int=None):
        self.l = None
        self.mat = None
        self.tessmat = None 
        if l != None and Rmat.shape == (3,3):
            self.l = l
            self.set_mat(Rmat, l)
            self.set_tesseral_mat(l)
    

    def set_mat(self, Rmat: np.array, l: int) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.")
            alpha, beta, gamma = R.from_matrix(Rmat).as_euler("zyz")
        smallwigner = self.__compute_small_wigner(l, beta)
        ms = np.arange(-l, l+1)
        # -1j or 1j ???
        expalpha = np.diag(np.exp(-1j * ms * alpha))
        expgamma = np.diag(np.exp(-1j * ms * gamma))
        self.mat = expgamma @ smallwigner @ expalpha
    

    def get_mat(self) -> np.array:
        if self.mat == None:
            print()
            print("****************** WARNING ******************")
            print("WIGNER D MODULE: Matrix has not yet been set.")
            print("*********************************************")
            print()
        return self.mat
    

    def set_tesseral_mat(self, l: int) -> None:
        Tmat = Tesseral.mat(l)
        self.tessmat = np.real(Tmat @ self.mat @ Tmat.transpose().conjugate())


    def get_tesseral_mat(self) -> np.array:
        if self.tesseralmat == None:
            print()
            print("********************** WARNING ***********************")
            print("WIGNER D MODULE: Tesseral Matrix has not yet been set.")
            print("******************************************************")
            print()
        return self.tesseralmat
    

    def __compute_small_wigner(self, l: int, beta: float) -> np.array:
        d = np.zeros((2*l+1, 2*l+1))
        for i, m1 in enumerate(range(-l, l+1)):
            for j, m2 in enumerate(range(-l, l+1)):
                d[i][j] = self.__compute_d_entry(l, m1, m2, beta)
        return d

    
    def __compute_d_entry(self, l: int, m1: int, m2: int, beta: float) -> float:
        prefac = np.sqrt(
            factorial(l + m1) * factorial(l - m1) * factorial(l + m2) * factorial(l - m2)
        )
        totsum = 0.0
        smin = max(0, m2-m1)
        smax = min(l+m2, l-m1)
        for s in range(smin, smax+1):
            num = (-1)**(m1 - m2 + s) * (np.cos(beta/2))**(2*l + m2 - m1 - 2*s) * (np.sin(beta/2))**(m1 - m2 + 2*s)
            denom = factorial(l + m2 - s) * factorial(s) * factorial(m1 - m2 + s) * factorial(l - m1 - s)
            totsum += num / denom
        return prefac * totsum
