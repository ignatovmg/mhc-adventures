import pandas as pd
import prody
import re
import numpy as np
import logging
import os
import tempfile
from glob import glob
from path import Path
from StringIO import StringIO
from subprocess import Popen, PIPE, STDOUT

from . import define
from . import utils
from . import atom_naming
from . import wrappers

_GDOMAINS_TABLE = pd.read_csv(define.TEMPLATE_MODELLER_DEFAULT_TABLE, index_col=0)

_GDOMAINS_DIR = define.GDOMAINS_DIR

_ALLELE_TABLE = pd.read_csv(define.ALLELE_SEQUENCES_CSV, sep=' ', index_col=1)


class BasePDB(object):
    _chain_order = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def __init__(self, pdb_file=None, pdb_list=None, ag=None, read_energy=False):
        self.energy_table = None

        if not pdb_file is None:
            self.ag = prody.parsePDB(pdb_file)
            if read_energy:
                self.energy_table = self.read_energy_from_pdb(pdb_file)

        elif not pdb_list is None:
            ag_first = prody.parsePDB(pdb_list[0])
            new_csets = []
            for f in pdb_list:
                ag = prody.parsePDB(f)
                assert (list(ag.getNames()) == list(ag_first.getNames()))
                new_csets.append(ag.getCoords())
            ag_first.setCoords(np.array(new_csets))
            self.ag = ag_first

        elif not ag is None:
            self.ag = ag.copy()

        else:
            raise ValueError('No molecules specified')

    def __getattr__(self, name):
        return self.ag.__getattribute__(name)

    def renumber(self, keep_resi=True, keep_chains=True):
        self.ag.setSerials(range(1, self.ag.numAtoms() + 1))

        if not keep_chains and self.ag.numChains() > len(self._chain_order):
            raise RuntimeError('Number of chains is larger, than the alphabet size')

        if not (keep_resi and keep_chains):
            for chaini, chain in enumerate(self.ag.iterChains()):
                if not keep_chains:
                    chain.setChid(self._chain_order[chaini])
                if not keep_resi:
                    for resi, res in enumerate(chain.iterResidues(), 1):
                        res.setResnum(resi)
        return self

    def copy(self):
        new = BasePDB(ag=self.ag)
        if not self.energy_table is None:
            new.energy_table = self.energy_table.copy()
        return new

    def _make_csets(self, csets):
        if csets is None:
            return range(self.ag.numCoordsets())
        if type(csets) == int:
            return [csets]
        if type(csets) == list:
            return csets
        raise TypeError('Wrong type of csets')

    def reduce(self, trim_first=True, csets=None):
        output = []
        natoms = -1
        csets = self._make_csets(csets)

        for i in csets:
            if trim_first:
                p_start = Popen([define.REDUCE_EXE, '-Quiet', '-Trim', '-'], stdin=PIPE, stdout=PIPE, stderr=STDOUT)
                p_finish = Popen([define.REDUCE_EXE, '-Quiet', '-FLIP', '-'], stdin=p_start.stdout, stdout=PIPE,
                                 stderr=STDOUT)
            else:
                p_start = Popen([define.REDUCE_EXE, '-Quiet', '-FLIP', '-'], stdin=PIPE, stdout=PIPE, stderr=STDOUT)
                p_finish = p_start

            prody.writePDBStream(p_start.stdin, self.ag, csets=i)
            p_start.stdin.close()

            output += ['MODEL%9i\n' % (i + 1)]
            reduced = []
            while p_finish.poll() is None:
                reduced = p_finish.stdout.readlines()

            natoms_cur = len(filter(lambda x: x.startswith('ATOM') or x.startswith('HETATM'), reduced))
            if i == csets[0]:
                natoms = natoms_cur
            elif natoms != natoms_cur:
                raise RuntimeError('Number of atoms in reduced model %i is different from the first model (%i, %i)' % (
                i, natoms_cur, natoms))

            output += reduced
            output += ['ENDMDL\n']

            status = p_finish.poll()

            if status != 0:
                logging.error('Called process returned ' + str(status))

        self.ag = prody.parsePDBStream(StringIO(''.join(output)))
        self.renumber()
        return output

    def save(self, pdb, **kwargs):
        if not pdb.endswith('.pdb'):
            raise RuntimeError('Filename must end with .pdb')

        ag = self.ag.copy()
        new_names = ['%4s' % ('%-3s' % a) if not ('0' <= a[0] <= '9') else '%-4s' % a for a in ag.getNames()]
        ag.setNames(new_names)

        prody.writePDB(pdb, ag, **kwargs)

    def save_sep(self, dirname, **kwargs):
        dirname = Path(dirname)
        dirname.mkdir_p()

        flist = []
        for cset in range(self.ag.numCoordsets()):
            fname = str(dirname.joinpath('%i.pdb' % cset))
            self.save(fname, csets=cset, **kwargs)
            flist.append(fname)

        return flist

    def prepare_pdb22_one_frame(self,
                                out_prefix,
                                cset=0,
                                rtf=define.RTF22_FILE,
                                prm=define.PRM22_FILE,
                                change_his=True,
                                remove_tmp=True,
                                patch_termini=True):

        pwd = Path.getcwd()

        try:
            pdb = Path(out_prefix + '_orig.pdb').abspath()
            rtf = Path(rtf).abspath()
            prm = Path(prm).abspath()

            basename = Path(pdb.basename().lower())
            dirname = pdb.dirname()
            dirname.chdir()

            if change_his:
                self.his_to_hsd()
            self.save(basename, csets=cset)

            call = [define.PDBPREP_EXE, basename]
            wrappers.shell_call(call)

            call = [define.PDBNMD_EXE, basename, '--rtf=%s' % rtf, '--prm=%s' % prm]
            # if patch_chains:
            #    call += ['--first', ','.join(['nter'] + [x.lower() for x in patch_chains])]
            #    call += ['--last',  ','.join(['cter'] + [x.lower() for x in patch_chains])]
            if patch_termini:
                call += ['--default-patch']

            call += ['?']
            wrappers.shell_call(call)

            nmin = Path(basename.stripext() + '_nmin.pdb')
            psf = Path(basename.stripext() + '_nmin.psf')
            wrappers.file_is_empty_error(nmin)

            outnmin = pwd.joinpath(out_prefix + '_nmin.pdb')
            outpsf = pwd.joinpath(out_prefix + '_nmin.psf')
            nmin.move(outnmin)
            psf.move(outpsf)

            if remove_tmp:
                files = basename.stripext() + '-*.????.pdb'
                call = ['rm', '-f'] + glob(files) + [basename]
                wrappers.shell_call(call)
        except:
            pwd.chdir()
            raise

        pwd.chdir()
        return outnmin, outpsf

    @staticmethod
    def _match_by_residue_position(pdb, ref):
        """
        Matches atom names to the reference residuewise
        """
        new_order = []
        natoms = 0
        for r1, r2 in zip(pdb.iterResidues(), ref.iterResidues()):
            # same residue
            logging.debug('%s - %s' % (r1, r2))
            assert (r1.getResname() == r2.getResname())
            atoms1 = r1.getNames()
            atoms2 = r2.getNames()

            # identical naming
            assert (set(atoms1) == set(atoms2))

            # non-redundant names
            assert (len(set(atoms1)) == len(atoms1))
            s = pd.Series(r1.getIndices(), atoms1)
            new_order += list(s[atoms2].values)

            natoms += len(atoms1)

        coords = pdb.getCoordsets()
        coords = coords[:, new_order, :]
        #for set_id in range(coords.shape[0]):
        #    coords[set_id] = coords[set_id][new_order]

        ref = ref.copy()
        ref.setCoords(coords)
        return ref

    def prepare_pdb22(self, out_prefix, csets=None, **kwargs):
        csets = self._make_csets(csets)

        nmin, psf = self.prepare_pdb22_one_frame(out_prefix, **kwargs)
        nmin_ag = prody.parsePDB(nmin)

        if len(csets) == 1:
            self.ag = nmin_ag
            self.save(nmin)
            return self

        if nmin_ag.numAtoms() == self.ag.numAtoms():
            if list(nmin_ag.getNames()) != list(self.ag.getNames()):
                nmin_ag = self._match_by_residue_position(self.ag, nmin_ag)
            else:
                nmin_ag.setCoords(self.ag.getCoordsets())
        else:
            logging.info('Molecule was altered during preparation, preparing each frame separately')
            new_csets = []
            for cset in csets:
                nmin_frame, psf_frame = self.prepare_pdb22_one_frame(out_prefix + '-%i-tmp' % cset, cset=0, **kwargs)
                ag_frame = prody.parsePDB(nmin_frame)
                assert (list(nmin_ag.getNames()) == list(ag_frame.getNames()))

                new_csets.append(ag_frame.getCoords())
                nmin_frame.remove()
                psf_frame.remove()

            nmin_ag.setCoords(np.array(new_csets))

        self.ag = nmin_ag
        self.save(nmin)
        return self

    def to_rosetta(self):
        new_names = []
        for rname, aname in zip(self.ag.getResnames(), self.ag.getNames()):
            if rname in ['HSD', 'HSE', 'HSP']:
                rname = 'HIS'
            k = (rname, aname)

            if k in atom_naming.atom_alias_ros:
                rname, aname = atom_naming.atom_alias_ros[(rname, aname)]
            new_names.append((rname, aname))

        new_names = zip(*new_names)
        self.ag.setResnames(new_names[0])
        self.ag.setNames(new_names[1])
        return self

    def from_rosetta(self):
        new_names = [atom_naming.atom_alias_ros_reverse.get((r, a), (r, a)) for r, a in
                     zip(self.ag.getResnames(), self.ag.getNames())]

        new_names = zip(*new_names)
        self.ag.setResnames(new_names[0])
        self.ag.setNames(new_names[1])
        return self

    def his_to_hsd(self):
        ag = self.ag
        new_resnames = []
        if 'HIS' in ag.getResnames():
            reslist = ag.iterResidues()
            for res in reslist:
                new_resn = res.getResname()
                if res.getResname() == 'HIS':
                    anames = res.getNames()
                    if 'HD1' in anames:
                        if 'HE2' in anames:
                            new_resn = 'HSP'
                        else:
                            new_resn = 'HSD'
                    else:
                        new_resn = 'HSE'
                new_resnames += [new_resn] * res.numAtoms()
            ag.setResnames(new_resnames)
        return self

    def hsd_to_his(self):
        sel = self.ag.select('resname HSD HSE HSP')
        if sel:
            sel.setResnames('HIS')
        return self

    def read_energy_from_pdb(self, pdb_file):
        ptn = re.compile(r'^REMARK\s+(\S+)\s+(\S+)\:\s+(-{0,1}[0-9na]+\.{0,1}[0-9]*)\s+$')
        cols = ['MODEL_ID', 'LINE']
        with open(pdb_file, 'r') as f:
            # get the names of all recorded energy fields from the first model
            for line in f:
                m = ptn.match(line)
                if m:
                    cols.append((m.group(1) + '_' + m.group(2)).upper())
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    break

            # run through all the models and record energy values
            f.seek(0)
            rows = []
            mdli = 1
            for linei, line in enumerate(f):
                if line.startswith('MODEL'):
                    row = [mdli, linei]
                    mdli += 1
                    continue

                m = ptn.match(line)
                if m:
                    row.append(float(m.group(3)))

                if line.startswith('END'):
                    rows.append(row)
                    row = []

        energy_table = pd.DataFrame(rows, columns=cols)

        nmodels = self.ag.numCoordsets()
        if energy_table.shape[0] != nmodels:
            raise RuntimeError('Number of energy values is different from the number of models (%i != %i)' %
                               (energy_table.shape[0], nmodels))

        return energy_table

    def calc_rmsd_to_frame(self, frame, align=False, sel='all'):
        ag = self.ag.copy()
        ag.setACSIndex(frame)
        if align:
            prody.alignCoordsets(ag.select(sel))
        return prody.calcRMSD(ag.select(sel))

    def calc_rmsd_matrix_intra(self, align=False, sel='all'):
        ag = self.ag.copy()
        rmsd = []
        for i in range(ag.numCoordsets()):
            ag.setACSIndex(i)
            if align:
                prody.alignCoordsets(ag.select(sel))
            rmsd.append([prody.calcRMSD(ag.select(sel))])
        rmsd = np.concatenate(rmsd)
        return rmsd

    def add_mol(self, mol, keep_chains=False, keep_resi=False):
        m1 = self
        m2 = mol
        assert(m1.ag.numCoordsets() == m2.ag.numCoordsets())

        buf = StringIO()
        for i in range(m1.ag.numCoordsets()):
            buf.write('MODEL\n')
            prody.writePDBStream(buf, m1.ag, csets=i)
            prody.writePDBStream(buf, m2.ag, csets=i)
            buf.write('ENDMDL\n')

        buf.seek(0)
        joint = prody.parsePDBStream(buf)
        buf.close()

        joint = BasePDB(ag=joint)
        joint.renumber(keep_chains=keep_chains, keep_resi=keep_resi)
        return joint

    def __add__(self, mol):
        return self.add_mol(mol)


class MHCIAllele(object):
    def __init__(self, allele):
        self.allele = allele

    @property
    def allele_seq(self):
        return _ALLELE_TABLE.loc[self.allele, 'sequence']

    @property
    def allele_pseq_nielsen(self):
        return _ALLELE_TABLE.loc[self.allele, 'pseudo_nielsen']

    @property
    def allele_pseq_custom(self):
        return _ALLELE_TABLE.loc[self.allele, 'pseudo_custom']

    def allele_pseq_nielsen_compute(self):
        return utils.get_pseudo_sequence(self.allele_seq, utils.nielsen_ref_seq, utils.nielsen_residue_set)

    def allele_pseq_custom_compute(self):
        return utils.get_pseudo_sequence(self.allele_seq, utils.custom_ref_seq, utils.contacting_set)


class ProcessedPDB(BasePDB):
    def __init__(self, pdb_id, pdb_file):
        BasePDB.__init__(self, pdb_file)

        self.pdb_id = pdb_id
        self.pdb_seq = ''
        self.seq_len = 0


class MHCIPDB(ProcessedPDB, MHCIAllele):
    def __init__(self, pdb_id):
        if pdb_id not in _GDOMAINS_TABLE.index:
            raise RuntimeError('PDB ID %s in not in Gdomains table' % pdb_id)

        pdb_file = Path(_GDOMAINS_DIR).joinpath(pdb_id + '_mhc_ah.pdb')
        ProcessedPDB.__init__(self, pdb_id, pdb_file)

        self._line = _GDOMAINS_TABLE.loc[pdb_id, :]
        self.pdb_seq = self._line['mhc_seq']
        self.seq_len = len(self.pdb_seq)
        MHCIAllele.__init__(self, self._line['allele'])

    @property
    def pdb_pseq_nielsen(self):
        return self._line['pseudo_nielsen']

    @property
    def pdb_pseq_custom(self):
        return self._line['pseudo_custom']

    def pdb_pseq_nielsen_compute(self):
        """
        Get Nielsen pseudo sequence (34aa)
        :returns: Tuple(sequence, residue_id_list)
        """
        if self.pdb_seq is None:
            return RuntimeError('Sequence is empty')
        return utils.get_pseudo_sequence(self.pdb_seq, utils.nielsen_ref_seq, utils.nielsen_residue_set)

    def pdb_pseq_custom_compute(self):
        """
        Get custom pseudo sequence
        :returns: Tuple(sequence, residue_id_list)
        """
        if self.pdb_seq is None:
            return RuntimeError('Sequence is empty')
        return utils.get_pseudo_sequence(self.pdb_seq, utils.custom_ref_seq, utils.contacting_set)


class PeptidePDB(ProcessedPDB):
    def __init__(self, pdb_id):
        if pdb_id not in _GDOMAINS_TABLE.index:
            raise RuntimeError('PDB ID %s in not in Gdomains table' % pdb_id)

        pdb_file = Path(_GDOMAINS_DIR).joinpath(pdb_id + '_pep_ah.pdb')
        super(PeptidePDB, self).__init__(pdb_id, pdb_file)

        _line = _GDOMAINS_TABLE.loc[pdb_id, :]
        self.pdb_seq = _line['peptide']
        self.seq_len = len(self.pdb_seq)


class MultiModelPDB(BasePDB):
    def __init__(self, pdb_file):
        BasePDB.__init__(self, pdb_file)
