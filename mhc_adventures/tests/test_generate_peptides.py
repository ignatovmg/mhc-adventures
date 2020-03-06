import unittest
import prody
import numpy as np
import pytest
import itertools


from ..mhc_peptide import BasePDB
from ..sampling.generate_peptides import PeptideSampler
from .. import utils
from ..helpers import isolate, isolated_filesystem


@pytest.fixture()
def default_mhc():
    return utils.load_gdomains_mhc('1ao7')


@pytest.fixture()
def default_pep():
    return utils.load_gdomains_peptide('1ao7')


@isolate
def test_instantiate_with_seq():
    sampler = PeptideSampler('ADCHTRTAC')
    assert sampler.pep.numAtoms() > 10


@isolate
def test_instantiate_with_short_seq():
    with pytest.raises(RuntimeError):
        PeptideSampler('ADCH')


@isolate
def test_instantiate_with_long_seq():
    with pytest.raises(RuntimeError):
        PeptideSampler('ADCHLKKKKKKKKKKKK')


@isolate
def test_instantiate_with_wrong_letters_seq():
    with pytest.raises(RuntimeError):
        PeptideSampler('ADCHLBBKK')


@isolate
def test_instantiate_with_pdb():
    prody.writePDB('pep.pdb', utils.load_gdomains_peptide('1ao7'))
    sampler = PeptideSampler(pep='pep.pdb')
    assert sampler.pep.numAtoms() > 10


@isolate
def test_instantiate_with_pep_and_mhc():
    prody.writePDB('pep.pdb', utils.load_gdomains_peptide('1ao7'))
    prody.writePDB('mhc.pdb', utils.load_gdomains_mhc('1ao7'))
    sampler = PeptideSampler(pep='pep.pdb', rec='mhc.pdb')
    assert sampler.pep.numAtoms() > 10
    assert sampler.rec.numAtoms() > 100


@isolate
def test_instantiate_with_seq_and_custom_template():
    prody.writePDB('template.pdb', utils.load_gdomains_peptide('1ao7'))
    sampler = PeptideSampler('ADCHTRTAC', custom_template='template.pdb')
    assert sampler.pep.numAtoms() > 10


@pytest.mark.parametrize('nsamples', [1, 10, 100, 1000])
def test_generate_simple(nsamples):
    with isolated_filesystem():
        sampler = PeptideSampler(pep=utils.load_gdomains_peptide('1ao7'))
        sampler.generate_peptides(nsamples, 1, 0.3, 123)
        assert sampler.brikard.numCoordsets() == nsamples


@isolate
def test_generate_with_template():
    prody.writePDB('template.pdb', utils.load_gdomains_peptide('1ao7'))
    sampler = PeptideSampler('ADCHTRTAC', custom_template='template.pdb')
    sampler.generate_peptides(10, 1, 0.2, 123)
    assert sampler.brikard.numCoordsets() == 10


# TODO: Brikard fails on '3rwd' with the following error
# Error termination. Backtrace:
# ...
# At line 479 of file manipulate_chain.f90
# Fortran runtime error: Index '3031' of dimension 1 of array 'chain' above upper bound of 3030
@pytest.mark.parametrize('pep,rec', itertools.product(['1a1m', '1t22', '2bvo'], ['1a1m', '1t22', '2bvo']))
def test_generate_with_rec(pep, rec):
    with isolated_filesystem():
        sampler = PeptideSampler(pep=utils.load_gdomains_peptide(pep), rec=utils.load_gdomains_mhc(rec))
        sampler.generate_peptides(10, 1, 0.2, 123)
        assert sampler.brikard.numCoordsets() == 10


# check that receptor is fixed by default during sampling
def test_generate_receptor_fixed(default_mhc, default_pep):
    with isolated_filesystem():
        sampler = PeptideSampler(pep=default_pep, rec=default_mhc)
        sampler.generate_peptides(10, 1, 0.2, 123)
        assert sampler.brikard.numCoordsets() == 10
        rec_fixed = sampler.brikard.select('chain A')
        assert np.all(rec_fixed.getCoordsets(0) == rec_fixed.getCoordsets(1))


# check that receptor is flexible with sample_resi_within parameter set
def test_generate_receptor_flexible(default_mhc, default_pep):
    with isolated_filesystem():
        sampler = PeptideSampler(pep=default_pep, rec=default_mhc)
        sampler.generate_peptides(10, 1, 0.2, 123, sample_resi_within=7)
        assert sampler.brikard.numCoordsets() == 10
        rec_flex = sampler.brikard.select('chain A')
        assert np.any(rec_flex.getCoordsets(0) != rec_flex.getCoordsets(1))


@pytest.mark.parametrize('radius', range(1, 7, 2))
def test_generate_receptor_variable_radius(default_mhc, default_pep, radius):
    sampler = PeptideSampler(pep=default_pep, rec=default_mhc)
    sampler.generate_peptides(10, 1, 0.2, 123, sample_resi_within=radius)
    assert sampler.brikard.numCoordsets() == 10
