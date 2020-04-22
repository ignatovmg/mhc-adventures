import os
import logging
from path import Path

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

# logging
LOGGING_CONF = os.path.join(os.path.dirname(PACKAGE_ROOT), 'logging.conf')
logger = logging.getLogger('debug')

# directories
SRC_DIR = os.path.join(PACKAGE_ROOT, 'source')
BIN_DIR = os.path.join(PACKAGE_ROOT, 'executables')
MOL_PRM_DIR = os.path.join(PACKAGE_ROOT, 'mol-prms')
PDB_PREP_DIR = os.path.join(BIN_DIR, 'pdb_prep')
GRID_PRM_DIR = os.path.join(PACKAGE_ROOT, 'grid-prms')
TEST_DIR = os.path.join(PACKAGE_ROOT, 'data/test_data')
PEPTIDE_TEMPLATES_DIR = os.path.join(PACKAGE_ROOT, 'ptemplates')

# parameters
RTF22_FILE = os.path.join(MOL_PRM_DIR, 'top_all22_prot_changed_atom_names.rtf')
PRM22_FILE = os.path.join(MOL_PRM_DIR, 'par_all22_prot.prm')
ATOM_PROPERTY22_FILE = os.path.join(GRID_PRM_DIR, 'props_new.csv')
ATOM_TYPES22_FILE = os.path.join(GRID_PRM_DIR, 'types_new.csv')
ATOM_PROPERTY_DEFAULT = ATOM_PROPERTY22_FILE
ATOM_TYPES_DEFAULT = ATOM_TYPES22_FILE

# executables
PDBPREP_EXE = os.path.join(PDB_PREP_DIR, 'pdbprep.pl')
PDBNMD_EXE = os.path.join(PDB_PREP_DIR, 'pdbnmd.pl')
PSFGEN_EXE = os.path.join(PDB_PREP_DIR, 'psfgen')
NMIN_EXE = os.path.join(PDB_PREP_DIR, 'nmin')
GRID_EXE = os.path.join(BIN_DIR, 'property_grid')

# misc
GDOMAINS_DIR = '/datasets/gdomains'

GDOMAINS_ADD = os.path.join(PACKAGE_ROOT, 'additional')
TEMPLATE_MODELLER_DEFAULT_TABLE = os.path.join(GDOMAINS_ADD, 'gdomains-complete.csv')
ALLELE_SEQUENCES_CSV = os.path.join(GDOMAINS_ADD, 'allele_sequences_reduced.csv')
#TEMPLATE_MODELLER_DEFAULT_TABLE = '/gpfs/projects/KozakovGroup/mhc_learning/analysis/gdomains-complete.csv'
#ALLELE_SEQUENCES_CSV = '/gpfs/projects/KozakovGroup/mhc_learning/analysis/allele_sequences_reduced.csv'

CHIMERA_EXE = Path('/usr/local/bin/chimera')

# brikard
BRIKARD_DIR = Path('/gpfs/software/brikard_cash/')
BRIKARD_LIB = BRIKARD_DIR / 'lib'
BRIKARD_EXE = BRIKARD_DIR / 'bin/brikard'
ASSEMBLE_EXE = BRIKARD_DIR / 'bin/assemble'
MISSING_LOOPS_EXE = BRIKARD_DIR / 'bin/missing_loops'

# external
REDUCE_EXE = '/gpfs/projects/KozakovGroup/software/reduce.3.23.130521.linuxi386'
SCWRL_EXE = '/gpfs/software/scwrl4/Scwrl4'

