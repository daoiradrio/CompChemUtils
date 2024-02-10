import numpy as np
import matplotlib.pyplot as plt



def printmat(mat: np.array) -> None:
    print()
    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
                print(f"{mat[row,col]:15.8f}", end="\t")
        print()
    print()


def printvec(vec: np.array) -> None:
    print()
    for val in vec:
        print(f"{val:15.8f}")
    print()



def plotmat(mat: np.array) -> None:
    fig, ax = plt.subplots(1,1)
    ax.imshow(mat, vmin=-1, vmax=1, cmap="PiYG")
    for i in range(mat.shape[1]):
        for j in range(mat.shape[1]):
            ax.text(i, j, "%.2f" % mat[j,i], ha="center", va="center", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.hlines(np.arange(1, mat.shape[0])-0.5, -0.5, mat.shape[0]-0.5, color="black", linestyle="--", linewidth=1)
    ax.vlines(np.arange(1, mat.shape[0])-0.5, -0.5, mat.shape[0]-0.5, color="black", linestyle="--", linewidth=1)
    plt.show()