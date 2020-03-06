import unittest
import prody
import pytest

from ..mhc_peptide import BasePDB
from .. import utils, define, wrappers
from ..wrappers import isolate


def test_renumber_residues():
    mhc = utils.load_gdomains_mhc('1ao7')
    base = BasePDB(ag=mhc)
    print(base.renumber_residues().ag)


#def test_add_hydrogens(self):
#    mhc = utils.load_gdomains_mhc('1ao7')
#    base = BasePDB(ag=mhc)
#    base.add_hydrogens()


def test_his_to_hsd():
    mhc = utils.load_gdomains_mhc('1ao7')
    base = BasePDB(ag=mhc)
    base.his_to_hsd()


@isolate
def test_prepare_pdb22_one_frame():
    with wrappers.isolated_filesystem():
        mhc = utils.load_gdomains_mhc('1ao7')
        base = BasePDB(ag=mhc)
        pdb, psf = base._prepare_pdb22_one_frame('prepared')
        ag = prody.parsePDB(pdb)
