import numpy as np



def printmat(mat: np.array) -> None:
    print()
    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
                print(f"{mat[row][col]:15.8f}", end="\t")
        print()
    print()


def printvec(vec: np.array) -> None:
    print()
    for val in vec:
        print(f"{val:15.8f}")
    print()