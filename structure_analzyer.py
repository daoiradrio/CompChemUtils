



class StructureAnalyer():

    def __init__(self):
        self.cycle_bonds = None

    
    def find_cycles(self, bond_dict: dict):
        atoms = list(bond_dict.keys())
        start = atoms[0]
        status = {atom: "UNKNOWN" for atom in atoms}
        self.__cycle_detection(bond_dict, start, start, status)
    

    def __cycle_detection(self, bond_dict: dict, start: str, last: str, status: dict, ancestors: dict = {}):
        # base case 1: node done or terminal atom
        if status[start] == "KNOWN" or get_element(start) in ["H", "F", "Cl", "Br", "I"]:
            return
        # base case 2: ring found (visited node again on a path where it has already been visited)
        elif status[start] == "VISITED":
            # loop over all bond in list of rotatable bonds
            for torsion in self.central_torsions:
                # if bond is identical with first bond of current cycle delete it from list of
                # rotatable bonds
                if (torsion.atom1 == start and torsion.atom2 == last) or (torsion.atom2 == start and torsion.atom1 == last):
                    self.central_torsions.remove(torsion)
                    # if bond has been found in list break loop
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
                for torsion in self.central_torsions:
                    # if current and next form bond which is in list of rotatable bond delete it from the list
                    if (torsion.atom1 == before and torsion.atom2 == cur) or (torsion.atom2 == before and torsion.atom1 == cur):
                        self.central_torsions.remove(torsion)
                        # if bond has been found in list break loop
                        break
        # case: node not visited yet
        else:
            # store last visited node
            ancestors[start] = last
            # mark current node as visited
            status[start] = "VISITED"
            # visit all neighbouring nodes of current node in sence of a dfs
            for bond_partner in bond_partners[start]:
                # only traverse through chain, ignore terminal atoms
                if not get_element(bond_partner) in ["H", "F", "Cl", "Br", "I"]:
                    # no jumping forth and back between current and last visited node
                    if not bond_partner == ancestors[start]:
                        self._cycle_detection(bond_partners, bond_partner, start, status, ancestors)
            status[start] = "KNOWN"
