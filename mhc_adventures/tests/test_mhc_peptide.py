import unittest
import prody
import pytest
import numpy as np

from ..mhc_peptide import BasePDB
from .. import utils, define
from ..helpers import isolate


@pytest.fixture()
def default_mhc():
    return utils.load_gdomains_mhc('1ao7')


@pytest.fixture()
def default_pep():
    return utils.load_gdomains_peptide('1ao7')


@pytest.fixture()
def renum_mhc():
    return BasePDB(ag=utils.load_gdomains_mhc('1ao7')).renumber(keep_resi=False).ag


@pytest.fixture()
def renum_pep():
    return BasePDB(ag=utils.load_gdomains_peptide('1ao7')).renumber(keep_resi=False).ag


def test_renumber_residues():
    mhc = utils.load_gdomains_mhc('1ao7')
    BasePDB(ag=mhc).renumber()


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
    mhc = utils.load_gdomains_mhc('1ao7')
    base = BasePDB(mhc)
    pdb, psf = base._prepare_pdb22_one_frame('prepared')
    prody.parsePDB(pdb)


def test_merge(default_mhc, default_pep):
    ag = BasePDB(default_mhc).add_mol(BasePDB(default_pep)).ag
    assert ag.numChains() == 2


def test_merge_addition(default_mhc, default_pep):
    ag = (BasePDB(default_mhc) + (BasePDB(default_pep))).ag
    assert ag.numChains() == 2


def test_merge_serials(default_mhc, default_pep):
    ag = BasePDB(default_mhc).add_mol(BasePDB(default_pep)).ag
    assert np.all(ag.getSerials() == np.arange(1, ag.numAtoms() + 1))


def test_merge_fail_diff_numcoordsets(default_mhc, default_pep):
    with pytest.raises(RuntimeError):
        default_mhc.setCoords(np.stack([default_mhc.getCoords()]*2))
        BasePDB(default_mhc).add_mol(BasePDB(default_pep))


def test_merge_keep_resi_true(renum_mhc, renum_pep):
    ag = BasePDB(renum_mhc).add_mol(BasePDB(renum_pep), keep_resi=True).ag
    assert len(set(ag.getResnums())) == 181


def test_merge_keep_resi_false(renum_mhc, renum_pep):
    ag = BasePDB(renum_mhc).add_mol(BasePDB(renum_pep), keep_resi=False).ag
    assert len(set(ag.getResnums())) == 181 + 9


def test_merge_same_chains_keep_chains_false(renum_mhc, renum_pep):
    renum_pep.setChids('A')
    ag = BasePDB(renum_mhc).add_mol(BasePDB(renum_pep)).ag
    assert ag.numChains() == 2


def test_merge_same_chains_keep_chains_true(renum_mhc, renum_pep):
    renum_pep.setChids('A')
    ag = BasePDB(renum_mhc).add_mol(BasePDB(renum_pep), keep_chains=True).ag
    assert ag.numChains() == 1


def test_merge_same_chains_keep_chains_and_resi_true_fail(renum_mhc, renum_pep):
    with pytest.raises(RuntimeError):
        renum_pep.setChids('A')
        BasePDB(renum_mhc).add_mol(BasePDB(renum_pep), keep_chains=True, keep_resi=True)
