import os
import numpy as np
from CompChemUtils.transform import Fock
from CompChemUtils.files import write_xyz_file



def get_dftb_HCS(rkfpath: str, orthogonalize: bool=False) -> (np.array, np.array, np.array):
    dftbrkf = KFFile(rkfpath)
    orbitals = dftbrkf.read_section("Orbitals")
    matrices = dftbrkf.read_section("Matrices")
    nmos = orbitals["nOrbitals"]
    C = orbitals["Coefficients(1)"]
    S = matrices["Data(1)"]
    H = matrices["Data(2)"]
    C = np.array(C).reshape((nmos,nmos), order="F")
    S = np.array(S).reshape((nmos,nmos), order="F")
    H = np.array(H).reshape((nmos,nmos), order="F")
    if orthogonalize:
        H, X = Fock.orthogonalize_basis(H, S)
        C = X.T @ C
        S = X @ S @ X
    return H, C, S



def calc_dftb_HCS(elems: list, coords: np.array, orthogonalize: bool=False) -> (np.array, np.array, np.array):
    write_xyz_file("inmol.xyz", elems, coords)
    settings = Settings()
    settings.input.ams.system.geometryfile = os.path.join(os.getcwd(), "inmol.xyz")
    settings.input.ams.Task = "SinglePoint"
    settings.input.DFTB.Model = "SCC-DFTB"
    settings.input.DFTB.ResourcesDir = "DFTB.org/3ob-3-1"
    settings.input.DFTB.StoreMatrices = "Yes"
    job = AMSJob(settings=settings)
    res = job.run()
    nmos = res.readrkf(file="dftb", section="Orbitals", variable="nOrbitals")
    S = res.readrkf(file="dftb", section="Matrices", variable="Data(1)")
    H = res.readrkf(file="dftb", section="Matrices", variable="Data(2)")
    C = res.readrkf(file="dftb", section="Orbitals", variable="Coefficients(1)")
    C = np.array(C).reshape((nmos,nmos), order="F")
    S = np.array(S).reshape((nmos,nmos), order="F")
    H = np.array(H).reshape((nmos,nmos), order="F")
    if orthogonalize:
        H, X = Fock.orthogonalize_basis(H, S)
        C = X.T @ C
        S = X @ S @ X
    return H, C, S



def calc_multiple_dftb_HCS(elemslist: list, coordslist: list, orthogonalize: bool=False) -> (list, list, list):
    settings = Settings()
    settings.input.ams.Task = "SinglePoint"
    settings.input.DFTB.Model = "SCC-DFTB"
    settings.input.DFTB.ResourcesDir = "DFTB.org/3ob-3-1"
    settings.input.DFTB.StoreMatrices = "Yes"
    jobs = []
    i = 0
    for elems, coords in zip(elemslist, coordslist):
        write_xyz_file(f"inmol{i}.xyz", elems, coords)
        settings.input.ams.system.geometryfile = os.path.join(os.getcwd(), f"inmol{i}.xyz")
        i += 1
        job = AMSJob(settings=settings)
        jobs.append(job)
    multijob = MultiJob(children=jobs)
    multijob.run()
    Hs = []
    Cs = []
    Ss = []
    for job in multijob.children:
        nmos = job.results.readrkf(file="dftb", section="Orbitals", variable="nOrbitals")
        S = job.results.readrkf(file="dftb", section="Matrices", variable="Data(1)")
        C = job.results.readrkf(file="dftb", section="Orbitals", variable="Coefficients(1)")
        H = job.results.readrkf(file="dftb", section="Matrices", variable="Data(2)")
        C = np.array(C).reshape((nmos,nmos), order="F")
        S = np.array(S).reshape((nmos,nmos), order="F")
        H = np.array(H).reshape((nmos,nmos), order="F")
        if orthogonalize:
            H, X = Fock.orthogonalize_basis(H, S)
            C = X.T @ C
            S = X @ S @ X
        Hs.append(H)
        Cs.append(C)
        Ss.append(S)
    return Hs, Cs, Ss



def reorder_rose_H(H: np.array, CS12: np.array, basis_set: str, elems: list, atoms_order: list) -> np.array:
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


def get_rose_H(rosepath: str) -> np.array:
    roserkf = KFFile(rosepath)
    data = roserkf.read_section("Data")
    basis_set = data["Basis Set Type"]
    atoms_order = data["Fragments and Atoms Order"]
    natoms = len(atoms_order)
    atoms_order = atoms_order[natoms:]
    nao1 = data["nao1"]
    nmo2 = data["nmo2"]
    CS12 = np.array(data["C_S12"]).reshape((nao1, nmo2))
    Hrose = np.array(data["IAO_Fock"]).reshape((nmo2, nmo2))
    Hrose = reorder_rose(Hrose, CS12, basis_set, elems, atoms_order)
    return Hrose
