import unittest
import prody

from ..mhc_peptide import BasePDB
from .. import utils, define, wrappers


class TestBasePDB(unittest.TestCase):
    def test_renumber_residues(self):
        mhc = utils.load_gdomains_mhc('1ao7')
        base = BasePDB(ag=mhc)
        print(base.renumber_residues().ag)

    #def test_add_hydrogens(self):
    #    mhc = utils.load_gdomains_mhc('1ao7')
    #    base = BasePDB(ag=mhc)
    #    base.add_hydrogens()

    def test_his_to_hsd(self):
        mhc = utils.load_gdomains_mhc('1ao7')
        base = BasePDB(ag=mhc)
        base.his_to_hsd()

    def test_prepare_pdb22_one_frame(self):
        with wrappers.isolated_filesystem():
            mhc = utils.load_gdomains_mhc('1ao7')
            base = BasePDB(ag=mhc)
            pdb, psf = base._prepare_pdb22_one_frame('prepared')
            ag = prody.parsePDB(pdb)




if __name__ == '__main__':
    unittest.main()
