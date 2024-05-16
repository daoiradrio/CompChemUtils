import os
from CompChemUtils.structure import Structure
from CompChemUtils.rmsd import RMSD



files = "testcases/"
try:
    f = open(os.path.join(files, "Alanin.xyz"))
    f.close()
except:
    files = "tests/testcases/"



def test_graph_based_rmsd():
    rmsd_threshold = 0.1
    rmsdobj = RMSD()
    water1 = os.path.join(files, "H2O.xyz")
    water2 = os.path.join(files, "H2O_different_order_rotated.xyz")
    ala1 = os.path.join(files, "Alanin.xyz")
    ala2 = os.path.join(files, "Alanin_different_order.xyz")
    pep1 = os.path.join(files, "Tyr-Ala-Trp.xyz")
    pep2 = os.path.join(files, "Tyr-Ala-Trp_different_order.xyz")
    watertest = rmsdobj.tight_rmsd(water1, water2) <= rmsd_threshold
    alatest = rmsdobj.tight_rmsd(ala1, ala2) <= rmsd_threshold
    peptest = rmsdobj.tight_rmsd(pep1, pep2) <= rmsd_threshold
    assert watertest and alatest and peptest
