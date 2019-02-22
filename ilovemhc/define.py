import os

# directories
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PACKAGE_ROOT, 'source')
BIN_DIR = os.path.join(PACKAGE_ROOT, 'executables')

MOL_PRM_DIR = os.path.join(PACKAGE_ROOT, 'mol-prms')
PDB_PREP_DIR = os.path.join(BIN_DIR, 'pdb_prep')
GRID_PRM_DIR = os.path.join(PACKAGE_ROOT, 'grid-prms')
TEST_DIR = os.path.join(PACKAGE_ROOT, 'data/test_data')

# parameters
RTF22_FILE = os.path.join(MOL_PRM_DIR, 'top_all22_prot.rtf')
PRM22_FILE = os.path.join(MOL_PRM_DIR, 'par_all22_prot.prm')
ATOM_PROPERTY22_FILE = os.path.join(GRID_PRM_DIR, 'props_new.csv')
ATOM_TYPES22_FILE = os.path.join(GRID_PRM_DIR, 'types.csv')

ATOM_PROPERTY_DEFAULT = ATOM_PROPERTY22_FILE
ATOM_TYPES_DEFAULT = ATOM_TYPES22_FILE

# executables
PDBPREP_EXE = os.path.join(PDB_PREP_DIR, 'pdbprep.pl')
PDBNMD_EXE = os.path.join(PDB_PREP_DIR, 'pdbnmd.pl')
GRID_EXE = os.path.join(BIN_DIR, 'property_grid')
REDUCE_EXE = '/gpfs/projects/KozakovGroup/reduce.3.23.130521.linuxi386'
SCWRL_EXE = '/gpfs/projects/KozakovGroup/software/scwrl4/Scwrl4'

# external
GDOMAINS_DIR = '/gpfs/projects/KozakovGroup/mhc_learning/gdomains'
PLIP_DIR = '/gpfs/projects/KozakovGroup/mhc_learning/analysis/plip-stable'
TEMPLATE_MODELLER_DEFAULT_TABLE = '/gpfs/projects/KozakovGroup/mhc_learning/analysis/gdomains-complete.csv'