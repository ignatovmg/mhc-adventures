import pytest
import prody

from .. import utils
from ..helpers import isolated_filesystem
from ..sampling.template_modelling import TemplateModeller
from ..mhc_peptide import BasePDB


@pytest.fixture()
def default_mhc():
    return utils.load_gdomains_mhc('1ao7')


@pytest.fixture()
def default_pep():
    return utils.load_gdomains_peptide('1ao7')


def test_modeller_init():
    TemplateModeller()


def test_modeller_create_scwrl(default_mhc):
    with isolated_filesystem():
        modeller = TemplateModeller()
        mhc_seq = default_mhc.calpha.getSequence()
        modeller.create_model('model.pdb', mhc_seq, 'ACCCCCRTYKI', add_h=False, prepare=False)

        ag = prody.parsePDB('model.pdb')
        assert ag.select('chain B and name CA').getSequence() == 'ACCCCCRTYKI'
        assert ag.select('chain A and name CA').getSequence() == mhc_seq


def test_modeller_create_scwrl_prepare(default_mhc):
    with isolated_filesystem():
        modeller = TemplateModeller()
        mhc_seq = default_mhc.calpha.getSequence()
        modeller.create_model('model.pdb', mhc_seq, 'ACCCCCRTYKI', add_h=False, prepare=True)

        # if HSE is not converted to HIS, prody converts it into S when fetching hte sequence
        ag = BasePDB('model.pdb').hsd_to_his().ag
        assert ag.select('chain B and name CA').getSequence() == 'ACCCCCRTYKI'
        assert ag.select('chain A and name CA').getSequence() == mhc_seq


def test_modeller_create_scwrl_add_h(default_mhc):
    with isolated_filesystem():
        modeller = TemplateModeller()
        mhc_seq = default_mhc.calpha.getSequence()
        modeller.create_model('model.pdb', mhc_seq, 'ACCCCCRTYKI', add_h=True, prepare=False)

        # if HSE is not converted to HIS, prody converts it into S when fetching hte sequence
        ag = prody.parsePDB('model.pdb')
        assert ag.select('chain B and name CA').getSequence() == 'ACCCCCRTYKI'
        assert ag.select('chain A and name CA').getSequence() == mhc_seq
