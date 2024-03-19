import warnings
import numpy as np
from math import factorial
from random import random
from scipy.spatial.transform import Rotation as R
import quaternionic
import spherical



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



class Fock:

    def __init__(self):
        pass

    
    @staticmethod
    def orthogonalize_basis(H: np.array, S: np.array) -> (np.array, np.array):
        '''
        nmos = S.shape[0]
        _, U = np.linalg.eig(S)
        U = np.matrix(U)
        s = np.matmul(U.H, np.matmul(S, U))
        for i in range(nmos):
            s[i,i] = 1.0 / np.sqrt(s[i,i])
        X = np.matmul(U, np.matmul(s, U.H))
        Hortho = np.matmul(X, np.matmul(H, X))
        return np.asarray(Hortho), X
        '''
        nmos = S.shape[0]
        _, U = np.linalg.eig(S)
        s = np.dot(U.transpose().conjugate(), np.dot(S, U))
        for i in range(nmos):
            s[i][i] = 1.0 / np.sqrt(s[i][i])
        X = np.dot(U, np.dot(s, U.transpose().conjugate()))
        Hortho = np.dot(U, np.dot(H, X))
        return H, X



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
    # PROBLEM IF DOT PRODUCT IS CLOSE TO 1 OR -1, MIGHT JUMP OUT OF DEFINITION DOMAIN OF 
    # ARCCOS DUE TO MACHINE PRECISION
    # IDEA:
    # if dot product close to 1 (tolerance) then matrix=I
    # else if close to -1 (tolerance) then rotation about vector orthogonal to refvec by angle pi
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
    
    @staticmethod
    def rand_rot_mat() -> np.array:
        cx, cy, cz = np.random.choice(np.arange(-1, 1.0001, 0.0001), 3)
        return Rx.mat(cx * np.pi) @ Ry.mat(cy * np.pi) @ Rz.mat(cz * np.pi)
    


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



class myWignerD:

    def __init__(self, Rmat: np.array=None, l: int=None):
        self.l = None
        self.mat = None
        self.tessmat = None
        self.p_can_to_cart = np.array([
                [0,0,1],
                [1,0,0],
                [0,1,0]
        ])
        if l != None and Rmat.shape == (3,3):
            self.l = l
            self.set_mats(Rmat, l)
            #self.mat = np.zeros((2*l+1, 2*l+1), dtype=np.complex128)
            #Q = R.from_matrix(Rmat).as_quat()
            #Robj = quaternionic.array(Q).normalized
            #wigner = spherical.Wigner(ell_min=self.l, ell_max=self.l)
            #Dmat = np.zeros((3,3), dtype=np.complex128)
            #wigner.D(Robj, out=self.mat)
            #printmat(self.mat)
            #self.set_tesseral_mat(l)
    

    def set_mats(self, Rmat: np.array, l: int, to_cart: bool=True) -> None:
        self.mat = np.zeros((2*l+1, 2*l+1), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.")
            gamma, beta, alpha = R.from_matrix(Rmat).as_euler("zyz")
        smallwigner = self.__compute_small_wigner(l, beta)
        ms = np.arange(-l, l+1)
        expalpha = np.diag(np.exp(-1j * ms * alpha))
        expgamma = np.diag(np.exp(-1j * ms * gamma))
        self.mat = expalpha @ smallwigner @ expgamma
        #for i, m1 in enumerate(ms):
        #    for j, m2 in enumerate(ms):
        #        self.mat[i][j] = np.exp(-1j * m1 * alpha) * smallwigner[i,j] * np.exp(-1j * m2 * gamma)
        self.set_tesseral_mat(l, to_cart)
    

    def get_mat(self) -> np.array:
        if self.mat == None:
            print()
            print("****************** WARNING ******************")
            print("WIGNER D MODULE: Matrix has not yet been set.")
            print("*********************************************")
            print()
        return self.mat
    

    def set_tesseral_mat(self, l: int, to_cart: bool=True) -> None:
        Tmat = Tesseral.mat(l)
        self.tessmat = np.real(Tmat @ self.mat @ Tmat.conjugate().transpose())
        # CURRENTLY ONLY FOR L=1
        if to_cart:
            self.tessmat = self.p_can_to_cart @ self.tessmat @ self.p_can_to_cart.T


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

    '''
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
    '''
    def __compute_d_entry(self, l, mp, m, beta):
        prefac = np.sqrt(factorial(l-m)*factorial(l+m)*factorial(l-mp)*factorial(l+mp))
        totsum = 0
        smin = max(0, -m-mp)
        smax = min(l-m, l-mp)
        for s in range(smin, smax+1):
            num = (-1)**l-m-s
            num *= np.cos(beta/2)**(m+mp+2*s)
            num *= np.sin(beta/2)**(l-mp-s)
            denom = factorial(s)
            denom *= factorial(l-mp-s)
            denom *= factorial(m+mp+s)
            denom *= factorial(l-m-s)
            totsum += num/denom
        return prefac*totsum
