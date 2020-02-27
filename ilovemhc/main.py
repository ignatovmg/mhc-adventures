""" Peptide Sampler Runner
Usage:
>>> python main.py -csv pepgen_3101.csv -wdir ../run -logfile log -rot 1 2 3 -vdw 0.0 0.1 -resi 1e-05 1e-06
>>> python main.py -csv pepgen_3101.csv -rot 1 2 3 -vdw 0.0 0.1 -resi 1e-05 1e-06
>>> python main.py -csv pepgen_3101.csv -logfile log -s
>>> python main.py -csv pepgen_3101.csv --filters $(cat filters.txt)
>>> python main.py -csv pepgen_3101.csv --filters_file filters.txt
>>> python main.py -csv pepgen_3101.csv --filters dir 2yez_GHAEEYGAETL
"""

from pathlib import Path
import pandas as pd
import logging
import random
import os
import time
from time import gmtime, strftime
import itertools
import shutil
import argparse
import sys
import generate_peptides as gp

class CustomFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


# TODO: parse file through LoadFromFile
# class LoadFromFile(argparse.Action):
#     def __call__(self, parser, namespace, values, option_string=None):
#         with values as f:
#             # parser.parse_args(f.read().split(), namespace)
#             return f.read().split()


def parse_args(args=None):
    """ Parse arguments """
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        formatter_class=CustomFormatter
    )

    g_program = parser.add_argument_group("Runner settings")
    g_program.add_argument(
        '-wdir',
        dest='wdir',
        metavar="Path",
        default=os.getcwd() / Path("run_" + strftime("%d%m%Y_%H%M%S", gmtime())),
        type=Path,
        help="Work directory path. Default values is \'current_dir\'+\'run_\'+\'timestamp\'"
    )
    g_program.add_argument(
        '-logfile',
        dest='logfile',
        metavar="String",
        default='pepgen.log',
        type=str,
        help="Log file name (if does not contain \'.log\' suffix - it will be added"
    )

    g_program.add_argument(
        '-rot',
        dest='rotamers',
        metavar="Int",
        default=[3],
        type=int,
        nargs='+',
        help="Rotamers values: one or more integer value, for each PeptideSampler will execute"
    )

    g_program.add_argument(
        '-vdw',
        dest='vdw',
        metavar="Float",
        default=[0.0],
        type=float,
        nargs='+',
        help="VDW values: one or more float value, for each PeptideSampler will execute"
    )

    g_program.add_argument(
        '-resi',
        dest='resi',
        metavar="Float",
        default=[0.00001],
        type=float,
        nargs='+',
        help="Residues on MHC within values: one or more float value, for each PeptideSampler will execute"
    )

    g_program.add_argument(
        '-samples',
        dest='samples',
        metavar="Int",
        default=10000,
        type=int,
        help="Number of samples per each params combination"
    )
    g_program.add_argument(
        '-csv',
        dest='csv',
        metavar="Path",
        type=str,
        required=True,
        help="Path to csv file with (peptide, mhc_path, pep_path, ...)"
    )

    g_program.add_argument(
        '--rec',
        '-rec',
        action='store_true',
        default=False,
        help="With receptor"
    )
    
    g_program.add_argument(
        '--tpl',
        '-tpl',
        metavar="Str",
        type=str,
        help="With receptor"
    )

    g_filters = parser.add_mutually_exclusive_group()
    g_filters.add_argument(
        '--filters',
        dest='filters',
        metavar="Str",
        default=None,
        type=str,
        nargs='+',
        help="HALP!"
    )
    g_filters.add_argument(
        '--filters_file',
        dest='filters',
        metavar="Str",
        default=None,
        type=str,
        help="HALP!"
    )

    g_logger = parser.add_mutually_exclusive_group()
    g_logger.add_argument(
        '--debug',
        '-d',
        action='store_true',
        default=False,
        help="enable debugging"
    )
    g_logger.add_argument(
        '--silent',
        '-s',
        action='store_true',
        default=False,
        help="disable logging"
    )

    return parser.parse_args(args)


def setup_logger(options):
    """ Configure logging """
    root = logging.getLogger("")
    root.setLevel(logging.WARNING)
    logger = logging.getLogger(options.logger_file.stem)
    logger.setLevel(logging.WARNING)
    logger.setLevel(options.debug and logging.DEBUG or logging.INFO)
    if not options.silent:
        handler = logging.FileHandler(str(options.logger_file))
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root.addHandler(handler)
        if options.debug:
            print "Debug mode"
        else:
            print "Info mode"
    else:
        print "Silent mode"
    return logger, handler


def make_logfile(dir_path, logfile_name):
    """Create log file in given directory and return filepath."""
    path = Path(dir_path, logfile_name)
    if not path.exists():
        path.parent.mkdir(parents=True)
        path.touch()
    return path


def make_filters(filters):
    if filters is not None and not isinstance(filters, list):
        with open(filters) as f:
            filters = f.read().split()
        f.close()
    return filters


if __name__ == '__main__':
    options = parse_args()
    options.logger_file = make_logfile(options.wdir, options.logfile)
    app_logger, app_handler = setup_logger(options)
    import runner
    runner.logger = app_logger
    gp.logger.handlers = [app_handler]
    options.filters = make_filters(options.filters)
    runner = runner.Runner(options)
    runner.generate()

