from CompChemUtils.structure import Structure



class StructureAnalyzer():

    def __init__(self, molecule: Structure):
        self.molecule = molecule
        self.cutable_bonds = []
    

    def compute_cutable_bonds(self) -> None:
        if self.molecule.bond_mat is None:
            self.molecule.set_graph()
        if self.molecule.bond_dict is None:
            self.molecule.set_graph(as_matrix=False)
        for i in range(self.molecule.natoms):
            for j in range(i+1, self.molecule.natoms):
                if self.molecule.bond_mat[i][j] == 1:
                    self.cutable_bonds.append((i,j))
        print(len(self.cutable_bonds))
        self.__find_cycles()
        print(len(self.cutable_bonds))

    
    def __find_cycles(self) -> None:
        atoms = list(self.molecule.bond_dict.keys())
        start = atoms[0]
        status = {atom: "UNKNOWN" for atom in atoms}
        self.__cycle_detection(start, start, status)
    

    def __cycle_detection(self, start: int, last: int, status: dict, ancestors: dict = {}, bonds: list=[]) -> None:
        # base case 1: node done or terminal atom
        if status[start] == "KNOWN" or self.molecule.elems[start] in ["H", "F", "Cl", "Br", "I"]:
            return
        # base case 2: ring found (visited node again on a path where it has already been visited)
        elif status[start] == "VISITED":
            for bond in self.cutable_bonds:
                if (bond[0] == start and bond[1] == last) or (bond[0] == last and bond[1] == start):
                    self.cutable_bonds.remove(bond)
                    break
            cur = last
            # get all atoms of current ring by backtracking
            # loop while full ring has not been traversed yet
            while cur != start:
                # get current atom in ring
                before = cur
                # get next atom in ring
                cur = ancestors[cur]
                # loop over all rotatable bonds
                for bond in self.cutable_bonds:
                    # if current and next form bond which is in list of rotatable bond delete it from the list
                    if (bond[0] == before and bond[1] == cur) or (bond[0] == before and bond[1] == cur):
                        self.cutable_bonds.remove(bond)
                        # if bond has been found in list break loop
                        break
        # case: node not visited yet
        else:
            # store last visited node
            ancestors[start] = last
            # mark current node as visited
            status[start] = "VISITED"
            # visit all neighbouring nodes of current node in sence of a dfs
            for bond_partner in self.molecule.bond_dict[start]:
                # only traverse through chain, ignore terminal atoms
                if not self.molecule.elems[bond_partner] in ["H", "F", "Cl", "Br", "I"]:
                    # no jumping forth and back between current and last visited node
                    if not bond_partner == ancestors[start]:
                        self.__cycle_detection(bond_partner, start, status, ancestors)
            status[start] = "KNOWN"
