import os

# directories
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PACKAGE_ROOT, 'source')
BIN_DIR = os.path.join(PACKAGE_ROOT, 'executables')

MOL_PRM_DIR = os.path.join(PACKAGE_ROOT, 'mol-prms')
PDB_PREP_DIR = os.path.join(BIN_DIR, 'pdb_prep')
GRID_PRM_DIR = os.path.join(PACKAGE_ROOT, 'grid-prms')
TEST_DIR = os.path.join(PACKAGE_ROOT, 'data/test_data')
PEPTIDE_TEMPLATES_DIR = os.path.join(PACKAGE_ROOT, 'brikard/ptemplates')

# parameters
RTF22_FILE = os.path.join(MOL_PRM_DIR, 'top_all22_prot_changed_atom_names.rtf')
PRM22_FILE = os.path.join(MOL_PRM_DIR, 'par_all22_prot.prm')
ATOM_PROPERTY22_FILE = os.path.join(GRID_PRM_DIR, 'props_new.csv')
ATOM_TYPES22_FILE = os.path.join(GRID_PRM_DIR, 'types_new.csv')
ATOM_PROPERTY_DEFAULT = ATOM_PROPERTY22_FILE
ATOM_TYPES_DEFAULT = ATOM_TYPES22_FILE

# external
GDOMAINS_DIR = '/gpfs/projects/KozakovGroup/mhc_learning/gdomains'
PLIP_DIR = '/gpfs/projects/KozakovGroup/mhc_learning/analysis/plip-stable'
BRIKARD_DIR = '/gpfs/projects/KozakovGroup/software/brikard'
BRIKARD_LIB = os.path.join(BRIKARD_DIR, 'lib')
ROSETTA_DIR = '/gpfs/projects/KozakovGroup/software/rosetta_src_2015.20.57849_bundle'
ROSETTA_DB = os.path.join(ROSETTA_DIR, 'main/database')
TEMPLATE_MODELLER_DEFAULT_TABLE = '/gpfs/projects/KozakovGroup/mhc_learning/analysis/gdomains-complete.csv'

# executables
PDBPREP_EXE = os.path.join(PDB_PREP_DIR, 'pdbprep.pl')
PDBNMD_EXE = os.path.join(PDB_PREP_DIR, 'pdbnmd.pl')
GRID_EXE = os.path.join(BIN_DIR, 'property_grid')
REDUCE_EXE = '/gpfs/projects/KozakovGroup/software/reduce.3.23.130521.linuxi386'
SCWRL_EXE = '/gpfs/projects/KozakovGroup/software/scwrl4/Scwrl4'
MINIMIZE_EXE = '/gpfs/projects/KozakovGroup/software/minimization-libmol2/build/minimize'
BRIKARD_EXE = os.path.join(BRIKARD_DIR, 'bin/brikard')
ASSEMBLE_EXE = os.path.join(BRIKARD_DIR, 'bin/assemble')
MISSING_LOOPS_EXE = os.path.join(BRIKARD_DIR, 'bin/missing_loops')
FLEXPEPDOCK_EXE = os.path.join(ROSETTA_DIR, 'main/source/bin/FlexPepDocking.linuxgccrelease')

# logging
LOGGING_CONF = os.path.join(os.path.dirname(PACKAGE_ROOT), 'logging.conf')
