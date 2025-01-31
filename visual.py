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
    _, ax = plt.subplots(1,1)
    ax.imshow(mat, vmin=-1, vmax=1, cmap="PiYG")
    for i in range(mat.shape[1]):
        for j in range(mat.shape[1]):
            ax.text(i, j, "%.2f" % mat[j,i], ha="center", va="center", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.hlines(np.arange(1, mat.shape[0])-0.5, -0.5, mat.shape[0]-0.5, color="black", linestyle="--", linewidth=1)
    ax.vlines(np.arange(1, mat.shape[0])-0.5, -0.5, mat.shape[0]-0.5, color="black", linestyle="--", linewidth=1)
    plt.show()



def show_mol(elems: list, coords: np.array, connectivity: np.array, return_plot: bool=False):

    num_atoms = len(elems)

    # map bond orders to different line widths in plot
    linewidths = {
        1: 2, 
        2: 6,
        3: 10
    }

    # map elements to colors as often seen in the literature
    atom_colors = {
        "H": "lightgrey",
        "C": "grey",
        "N": "blue",
        "O": "red",
        "F": "green"
    }

    # initialize plot
    ax = plt.axes(projection="3d")
    ax.set_axis_off()

    for i in range(num_atoms):

        coord1 = coords[i, :]

        for j in range(i+1, num_atoms):

            coord2 = coords[j, :]
            bond_order = connectivity[i][j]

            if bond_order:
                # bonds are plotted as simple lines, the higher the bond order the thicker the line
                ax.plot3D(
                    [coord1[0], coord2[0]],
                    [coord1[1], coord2[1]],
                    [coord1[2], coord2[2]],
                    linewidth=linewidths[bond_order],
                    color="black"
                )

    for elem, coord in zip(elems, coords):
        
        # plot atoms as thick dots and corresponding element color
        ax.scatter(coord[0], coord[1], coord[2], color=atom_colors[elem], s=250)
    
    # either return or show plot
    if return_plot:
        return ax
    else:
        plt.show()
