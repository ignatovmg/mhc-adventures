import unittest
import prody
import numpy as np

from ..mhc_peptide import BasePDB
from ..generate_peptides import PeptideSampler
from .. import utils


class TestPeptideSampler(unittest.TestCase):
    def test_instantiate_with_seq(self):
        sampler = PeptideSampler('ADCHTRTAC')
        self.assertGreater(sampler.pep.numAtoms(), 10)

    def test_instantiate_with_pdb(self):
        with utils.isolated_filesystem():
            prody.writePDB('pep.pdb', utils.load_gdomains_peptide('1ao7'))
            sampler = PeptideSampler(pep='pep.pdb')
            self.assertGreater(sampler.pep.numAtoms(), 10)

    def test_instantiate_with_pep_and_mhc(self):
        with utils.isolated_filesystem():
            prody.writePDB('pep.pdb', utils.load_gdomains_peptide('1ao7'))
            prody.writePDB('mhc.pdb', utils.load_gdomains_mhc('1ao7'))
            sampler = PeptideSampler(pep='pep.pdb', rec='mhc.pdb')
            self.assertGreater(sampler.pep.numAtoms(), 10)
            self.assertGreater(sampler.rec.numAtoms(), 100)

    def test_instantiate_with_seq_and_custom_template(self):
        with utils.isolated_filesystem():
            prody.writePDB('template.pdb', utils.load_gdomains_peptide('1ao7'))
            sampler = PeptideSampler('ADCHTRTAC', custom_template='template.pdb')
            self.assertGreater(sampler.pep.numAtoms(), 10)

    def test_generate_simple(self):
        with utils.isolated_filesystem():
            sampler = PeptideSampler(pep=utils.load_gdomains_peptide('1ao7'))
            sampler.generate_peptides(10, 1, 0.2, 123)
            self.assertEqual(sampler.brikard.numCoordsets(), 10)

    def test_generate_with_template(self):
        with utils.isolated_filesystem():
            prody.writePDB('template.pdb', utils.load_gdomains_peptide('1ao7'))
            sampler = PeptideSampler('ADCHTRTAC', custom_template='template.pdb')
            sampler.generate_peptides(10, 1, 0.2, 123)
            self.assertEqual(sampler.brikard.numCoordsets(), 10)

    def test_generate_with_rec(self):
        with utils.isolated_filesystem():
            sampler = PeptideSampler(pep=utils.load_gdomains_peptide('1ao7'), rec=utils.load_gdomains_mhc('1ao7'))
            sampler.generate_peptides(10, 1, 0.2, 123)
            self.assertEqual(sampler.brikard.numCoordsets(), 10)

    def test_receptor_sampling_fixed(self):
        # check that receptor is fixed by default during sampling
        with utils.isolated_filesystem():
            sampler = PeptideSampler(pep=utils.load_gdomains_peptide('1ao7'), rec=utils.load_gdomains_mhc('1ao7'))
            sampler.generate_peptides(10, 1, 0.2, 123)
            self.assertEqual(sampler.brikard.numCoordsets(), 10)
            rec_fixed = sampler.brikard.select('chain A')
            self.assertTrue(np.all(rec_fixed.getCoordsets(0) == rec_fixed.getCoordsets(1)))

    def test_receptor_sampling_flexible(self):
        # check that receptor is flexible with sample_resi_within parameter set
        with utils.isolated_filesystem():
            sampler = PeptideSampler(pep=utils.load_gdomains_peptide('1ao7'), rec=utils.load_gdomains_mhc('1ao7'))
            sampler.generate_peptides(10, 1, 0.2, 123, sample_resi_within=7)
            self.assertEqual(sampler.brikard.numCoordsets(), 10)
            rec_flex = sampler.brikard.select('chain A')
            self.assertTrue(np.any(rec_flex.getCoordsets(0) != rec_flex.getCoordsets(1)))


if __name__ == '__main__':
    unittest.main()