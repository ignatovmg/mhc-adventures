import unittest
import prody

from ..mhc_peptide import BasePDB
from ..generate_peptides import PeptideSampler
from .. import utils


class TestBasePDB(unittest.TestCase):
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
            prody.writePDB('template.pdb', utils.load_gdomains_peptide('1ao7'))
            sampler = PeptideSampler('ADCHTRTAC', custom_template='template.pdb')
            sampler.generate_peptides(10, 1, 0.2, 123)
            self.assertEqual(sampler.brikard.numCoordsets(), 10)


if __name__ == '__main__':
    unittest.main()