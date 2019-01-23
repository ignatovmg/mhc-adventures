import os

# directories
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PACKAGE_ROOT, 'src')
BIN_DIR = os.path.join(PACKAGE_ROOT, 'bin')

MOL_PRM_DIR = os.path.join(SRC_DIR, 'prepare_pdb/mol-prms')
PDB_PREP_DIR = os.path.join(SRC_DIR, 'prepare_pdb/pdbprep')
GRID_DIR = os.path.join(SRC_DIR, 'grid_maker')
TEST_DIR = os.path.join(PACKAGE_ROOT, 'data/test_data')
GDOMAINS_DIR = os.path.join('/gpfs/projects/KozakovGroup/mhc_learning/gdomains')

# parameters
RTF22_FILE = os.path.join(MOL_PRM_DIR, 'top_all22_prot.rtf')
PRM22_FILE = os.path.join(MOL_PRM_DIR, 'par_all22_prot.prm')
ATOM_PROPERTY22_FILE = os.path.join(GRID_DIR, 'props.csv')
ATOM_TYPES22_FILE = os.path.join(GRID_DIR, 'types.csv')

ATOM_PROPERTY_DEFAULT = ATOM_PROPERTY22_FILE
ATOM_TYPES_DEFAULT = ATOM_TYPES22_FILE

# executables
PDBPREP_EXE = os.path.join(PDB_PREP_DIR, 'pdbprep.pl')
PDBNMD_EXE = os.path.join(PDB_PREP_DIR, 'pdbnmd.pl')
GRID_EXE = os.path.join(BIN_DIR, 'property_grid')
REDUCE_EXE = '/gpfs/projects/KozakovGroup/reduce.3.23.130521.linuxi386'