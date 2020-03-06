import logging.config
from path import Path
import json

SRC_DIR = Path(__file__).abspath().dirname()
ROOT_DIR = SRC_DIR.dirname()

with open(ROOT_DIR / 'vars.json', 'r') as f:
    _VARS = json.load(f)

# logging
logger = logging.getLogger('console')

# parameters
RTF22_FILE = SRC_DIR / 'mol-prms' / 'top_all22_prot_changed_atom_names.rtf'
PRM22_FILE = SRC_DIR / 'mol-prms' / 'par_all22_prot.prm'

# misc
GDOMAINS_DIR = Path(_VARS['GDOMAINS_DIR'])
PLIP_DIR = Path(_VARS['PLIP_DIR'])
TEMPLATE_MODELLER_DEFAULT_TABLE = '/gpfs/projects/KozakovGroup/mhc_learning/analysis/gdomains-complete.csv'
ALLELE_SEQUENCES_CSV = '/gpfs/projects/KozakovGroup/mhc_learning/analysis/allele_sequences_reduced.csv'

# brikard
BRIKARD_DIR = Path(_VARS['BRIKARD_DIR'])
BRIKARD_LIB = BRIKARD_DIR / 'lib'
BRIKARD_EXE = BRIKARD_DIR / 'bin' / 'brikard'
ASSEMBLE_EXE = BRIKARD_DIR / 'bin' / 'assemble'
MISSING_LOOPS_EXE = BRIKARD_DIR / 'bin' / 'missing_loops'

# rosetta
ROSETTA_DIR = Path(_VARS['ROSETTA_DIR'])
ROSETTA_DB = ROSETTA_DIR / 'main' / 'database'
FLEXPEPDOCK_EXE = ROSETTA_DIR / 'main' / 'source' / 'bin' / 'FlexPepDocking.linuxgccrelease'

# external
REDUCE_EXE = '/gpfs/projects/KozakovGroup/software/reduce.3.23.130521.linuxi386'
SCWRL_EXE = '/gpfs/projects/KozakovGroup/software/scwrl4/Scwrl4'
MINIMIZE_EXE = _VARS['MIN_EXE']
CCMPRED_EXE = '/gpfs/projects/KozakovGroup/software/CCMpred/bin/ccmpred'
NNALIGN_EXE = '/gpfs/projects/KozakovGroup/software/nnalign-2.1/nnalign'
