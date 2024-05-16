from CompChemUtils.rmsd import RMSD



file1 = "/Users/dario/CompChemUtils/tests/testcases/Alanin.xyz"
file2 = "/Users/dario/CompChemUtils/tests/testcases/Alanin_different_order.xyz"
rmsd_analyzer = RMSD()
rmsd = rmsd_analyzer.tight_rmsd(file1, file2)
