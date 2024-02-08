import warnings
import numpy as np
from math import factorial
from scipy.spatial.transform import Rotation as R



#TODO
# - class templates for rotations, currently lots repetitions among Rx, Ry and Rz (vielleicht sogar für sämtliche Transformationen zusammen?)



class Rx():

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
        return np.dot(rotmat, coord)



class Ry():

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
        return np.dot(rotmat, coord)



class Rz():

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
        return np.dot(rotmat, coord)



class Tesseral():

    def __init__(self, l: int=None):
        self.mat = None
        if l != None:
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



class WignerD():

    def __init__(self, Rmat: np.array=None, l: int=None):
        self.mat = None
        if Rmat != None and l != None:
            self.set_mat(Rmat, l)
    

    def set_mat(self, Rmat: np.array, l: int) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.")
            alpha, beta, gamma = R.from_matrix(Rmat).as_euler("zyz")
        smallwigner = self.__compute_small_wigner(l, beta)
        ms = np.arange(-l, l+1)
        expalpha = np.diag(np.exp(-1j * ms * alpha))
        expgamma = np.diag(np.exp(-1j * ms * gamma))
        self.mat = expgamma @ small_wignerd @ expalpha
    

    def get_mat(self) -> np.array:
        if self.mat == None:
            print()
            print("****************** WARNING ******************")
            print("WIGNER D MODULE: Matrix has not yet been set.")
            print("*********************************************")
            print()
        return self.mat
    

    def __compute_small_wigner(self, l: int, beta: float) -> np.array:
        d = np.zeros((2*l+1, 2*l+1))
        for i, m1 in enumerate(range(-l, l+1)):
            for j, m2 in enumerate(range(-l, l+1)):
                d[i][j] = self.__compute_d_entry(l, m1, m2, beta)
        return d

    
    def __compute_d_entry(l: int, m1: int, m2: int, beta: float) -> float:
        prefac = np.sqrt(
            factorial(l + m1) * factorial(l - m1) * factorial(l + m2) * factorial(l - m2)
        )
        sum = 0.0
        smin = min(0, m2-m1)
        smax = max(l+m2, l-m1)
        for s in range(smin, smax+1):
            nom = (-1)**(m1 - m2 + s) * (np.cos(beta/2))**(2*l + m2 - m1 - 2*s) * (np.sin(beta/2))**(m1 - m2 + 2*s)
            denom = factorial(l + m2 - s) * factorial(s) * factorial(m1 - m2 + s) * factorial(l - m1 - s)
            sum += nom / denom
        return prefac * sum