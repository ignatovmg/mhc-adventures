from glob import glob
import numpy as np
import subprocess
from subprocess import Popen
from time import time
import itertools

import prody
from path import Path
from Bio.SeqUtils import seq1, seq3
from Bio.Alphabet.IUPAC import IUPACProtein

from . import define, wrappers
from .mhc_peptide import BasePDB

logger = define.logger


_assemble_production = '''Chain: two loops
# problem description
{brikard_lib}
# path to library (relative or absolute)
.
# output directory (default is .)
{output_dir}/mol.mol2
# loop, prefix for combined structure (can omit the .mol2 suffix)
C
# C: chain
# R: ring

# (blank): de novo
# R: relative refinement
# A: absolute refinement
0
# number of kinematic loops
1
# number of loops
{resin} {resic}
# beginning and end of loop
{pivots}
# pivot residue numbers [may be followed by (atom), e.g,, 1(CA) 2(N) 4(CB)]
{sampled}
# residue numbers for sampled residues; if blank, no residues are sampled
# [may be followed by (torsion1,...), e.g., 1(1,2) 2(1,3) 3(2)]

# backbone loop closure conditions (B = bond, A = angle, T = torsion):
#    B(n,n+1),         A(n-1,n,n+1),     A(n,n+1,n+2),
#    T(n-2,n-1,n,n+1), T(n-1,n,n+1,n+2), T(n,n+1,n+2,n+3)
# If the above are not specified, they will be defaulted/computed.
0 0.000000001 {vdw:.3f}
# imeth, acceptable probability, vdW fraction (= 1 - vdW overlap)
{N:d}
# max_sols: maximal solutions requested
0
#  debug level: -1: no output, 0: normal, larger values: lots of output!
100000000 100 {nrot:d}
# max_sample
# max_trials: for automatic operation, need copies of p1.pdb in p1_001.pdb---
#  -- p1_N.pdb where N = max_trials (3 digit decimal, 000 to 999 max)
#     but also p1.pdb, p2.pdb must be present (copies of px_001.pdb, x=1,2)
# rot_sample: number of resamplings of rotamers, to avoid clashes.
1 0.20
# initial Sobol seed, Monte Carlo eps_t fraction
#sBAS
sBA1
# String of one-character options to determine various behaviors:
#    create DCD files:   d
#    create LOOPS file:  L
#    create log file:    l
#    rotamers            T     (none)
#    torsion sampling:   s,r,q (Sobol_rama, Random_rama, Random)
#    energy scheme:      S,P,D (Seok, PLOP, Delaunay)
#    minimize energy:    m,M   (Minimize/write, Minimize/ingest)
#    Monte Carlo:        C
#    Ramachandran:       R
#    sterics:            B,A   (Backbone+, All)
'''


class PeptideSampler(object):
    """
    This can generate a peptide or read one from pdb and exhaustively
    sample it with fixed termini using BRIKARD, inverse kinematics algorithm.

    The receptor, such a MHC, can also be provided and the sidechains within
    a given threshold can be sampled as well.

    Usage:

    >>> sampler = PeptideSampler(pep_seq='ACACACACAC', rec='mhc.pdb')
    >>> sampler.generate_peptides(100, 3, 0.3, 123, sample_resi_within=6.0, auto_disu=True)

    This will create 100 peptide samples trying to produce 3 different rotamer sets for each
    backbone conformation, with 0.3 vdw threshold and random seed 123. Residues on MHC within
    6 A will be sampled, except for those involved in disulfide bonds.

    The result is saved to brikard.pdb and stored in sampler.brikard.
    """

    # TODO: add peptide preminimization

    _disu_dist = 2.5  # distance for disu bonds identification
    _vdw_min = 0.0  # min vdw threshold to accept during sampling
    _sampling_init_timeout = 5 * 60

    def __init__(self, pep_seq=None, pep=None, rec=None, wdir='.', prepare=True, custom_template=None):
        if not define.BRIKARD_EXE.exists():
            raise OSError('Brikard executable is missing ({})'.format(define.BRIKARD_EXE))

        self.wdir = Path(wdir)
        self.wdir.mkdir_p()
        self.prepare = prepare

        self._seq_file = self.wdir/'seq_file'
        self._scw_file = self.wdir/'pep_scw.pdb'
        self._mrg_file = self.wdir/'merged.pdb'
        self._input_mol2_file = self.wdir/'mol.mol2'
        self._input_pdb_file = self.wdir/'input.pdb'
        self._brikard_file = self.wdir/'brikard.pdb'
        self._a_file = self.wdir/'a.mhc'

        self.rec = rec
        if isinstance(self.rec, str):
            logger.info('Reading receptor')
            self.rec = prody.parsePDB(rec)
        if isinstance(self.rec, prody.AtomGroup) and prepare:
                logger.info('Preparing receptor')
                self.rec = BasePDB(ag=self.rec).prepare_pdb22('mhc').hsd_to_his().ag

        self.pep = pep
        self.pep_seq = pep_seq
        self.custom_template = None if custom_template is None else Path(custom_template).abspath()

        if pep_seq is not None:
            logger.info('Generating starting peptide from sequence')
            self._check_sequence(pep_seq)
            # make peptide starting conformation
            self._make_pep_from_seq()
        else:
            if isinstance(self.pep, str):
                logger.info('Reading peptide')
                self.pep = prody.parsePDB(self.pep)
            if isinstance(self.pep, prody.AtomGroup):
                self.pep_seq = BasePDB(ag=self.pep).get_sequence()
                if prepare:
                    logger.info('Preparing peptide')
                    self.pep = BasePDB(ag=self.pep).prepare_pdb22('pep').hsd_to_his().ag
            else:
                raise ValueError('Cannot load peptide')

        self.pep_len = self.pep.numResidues()

        self._tpl = None
        self._input = None
        self._disu_pairs = None

        # sampled peptides will be here
        self.brikard = None
        self.brikard_raw_file = None

    def clean(self):
        self._seq_file.remove_p()
        self._scw_file.remove_p()
        self._mrg_file.remove_p()
        self._input_mol2_file.remove_p()
        self._input_pdb_file.remove_p()
        self.wdir.joinpath('test.pdb').remove_p()
        self.wdir.joinpath('test0.pdb').remove_p()
        self.wdir.joinpath('test1.pdb').remove_p()
        self.wdir.joinpath('assemble.tmp').remove_p()
        if self.brikard_raw_file:
            self.brikard_raw_file.remove_p()

    @staticmethod
    def _check_sequence(pep_seq):
        if len(pep_seq) < 8 or len(pep_seq) > 14:
            raise RuntimeError('Peptides sequence cannot be longer than 14 or shorter than 8 residues')
        for a in pep_seq:
            if a not in IUPACProtein.letters:
                raise RuntimeError('Residue %s is not a standard amino acid' % a)

    def _generate_template(self):
        bbnames = ['C', 'O', 'CA', 'N', 'OXT']
        lgt = len(self.pep_seq)
        if self.custom_template is None:
            tpl = Path(define.PEPTIDE_TEMPLATES_DIR) / str(lgt) + 'mer.pdb'
        else:
            tpl = self.custom_template
        tpl = prody.parsePDB(tpl)

        for r, newname in zip(tpl.iterResidues(), self.pep_seq):
            r.setResname(seq3(newname).upper())
        tpl = tpl.select('name ' + ' '.join(bbnames)).copy()
        tpl.setChids('B')
        self._tpl = tpl

    def _make_scwrl_sequence_file(self):
        l = self._tpl.select('name CA').getResnames()
        r = []
        if self.rec is not None:
            r = self.rec.select('name CA').getResnames()

        seq = ''.join(list(map(seq1, r))).lower() + ''.join(list(map(seq1, l))).upper()
        self._seq_file.write_text(seq + '\n')

    def _generate_sidechains_scwrl(self):
        if not self.rec is None:
            rec = self.rec.copy()
            rec.setChids('A')
            lig = self._tpl.copy()
            rec = BasePDB(ag=rec)
            lig = BasePDB(ag=lig)
            merged = rec.add_mol(lig, keep_resi=False, keep_chains=True)
            merged.save(self._mrg_file)

            self._make_scwrl_sequence_file()
            call = [define.SCWRL_EXE, '-h', '-i', self._mrg_file, '-o', self._scw_file, '-s', self._seq_file]
        else:
            prody.writePDB(self._mrg_file, self._tpl)
            call = [define.SCWRL_EXE, '-h', '-i', self._mrg_file, '-o', self._scw_file]

        # scwrl wants rosetta hydrogen naming
        BasePDB(self._mrg_file).to_rosetta().save(self._mrg_file)

        wrappers.shell_call(call)

        pep = prody.parsePDB(self._scw_file)

        # extract peptide and renumber
        pep = BasePDB(ag=pep.select('chain B').copy()).renumber_residues(keep_resi=False).ag
        self.pep = pep

    def _make_pep_from_seq(self):
        """
        Create peptide from sequence and put it in self.pep
        """
        self._generate_template()
        self._generate_sidechains_scwrl()
        return self.pep

    def _run_brikard(self, N, resin, resic, nrot=1, vdw=0.3, seed=123, rec_resi_list=None, restrictions=None):
        # pivots = (resin + 1, (resin + resic) / 2, resic - 1)
        outdir = self.wdir
        oldpwd = Path.getcwd()
        outdir.chdir()

        self.brikard_raw_file = None
        self.brikard = None

        residues = range(resin, resic + 1)
        pivots = tuple(residues[3:6])

        sampled = set(range(resin + 1, resic))
        sampled = sorted(list(sampled - set(pivots)))
        pivots = '{0}(N) {0}(CA) {1}(N) {1}(CA) {2}(N) {2}(CA)'.format(*pivots)
        sampled = '%i(2) %s %i(1)' % (resin, " ".join([str(x) for x in sampled]), resic)

        try:
            wrappers.remove_files(glob('mol_000001*.pdb'))

            assemble_file_content = _assemble_production.format(brikard_lib=define.BRIKARD_LIB,
                                                                output_dir='.',
                                                                resin=resin,
                                                                resic=resic,
                                                                pivots=pivots,
                                                                sampled=sampled,
                                                                N=N,
                                                                nrot=nrot,
                                                                vdw=vdw)

            if rec_resi_list:
                assemble_file_content += '@ROTAMER_INCLUSIONS\n'
                assemble_file_content += ' '.join(map(str, rec_resi_list))

            if restrictions:
                assemble_file_content += '@RESTRICTIONS\n'
                for (resi, torsion), limits in sorted(restrictions.items()):
                    assemble_file_content += '{:d} {:d} {}\n'.format(resi, torsion, limits)

            a_file = self._a_file.basename()
            a_file.write_text(assemble_file_content)

            # run brikard
            wrappers.shell_call([define.ASSEMBLE_EXE, a_file])
            process = Popen(define.BRIKARD_EXE, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            process.communicate(str(seed).encode('utf-8'))

            noutputs = 0
            output = ''
            start_time = time()
            while output or (process.poll() is None):
                output = process.stdout.readline().decode('utf-8')
                if output:
                    logger.info(output.strip())
                    if 'accepted =' in output:
                        naccepted = int(output.split('accepted = ')[-1])
                        noutputs += 1
                        progress = naccepted / float(noutputs)

                        # Kill if progress is too slow
                        if noutputs > 50 and progress < 0.05:
                            logger.info("Progress is too slow: %f. Killing ..." % progress)
                            process.kill()
                        if naccepted > N:
                            logger.info("Brikard has generated %i (more than requested).." % naccepted)
                            # process.kill()

                # Kill if cannot initialize sampling in 5 mins
                if noutputs == 0:
                    if time() - start_time > self._sampling_init_timeout:
                        logger.info('Couldn\'t find the first lead in 5 minutes, decrease VDW penalty')
                        process.kill()

            process.wait()

            brikard_raw = glob('mol_*.pdb')
            if len(brikard_raw) == 0:
                logger.warning('No sampled structures found (mol_000001*.pdb)')
                oldpwd.chdir()
                return

            brikard_raw = brikard_raw[0]
            logger.info('Using %s' % brikard_raw)
            self.brikard_raw_file = outdir.joinpath(Path(brikard_raw))
            self.brikard = prody.parsePDB(brikard_raw)

        except Exception:
            oldpwd.chdir()
            raise

        oldpwd.chdir()

    def _pdb_to_mol2(self):
        """
        Convert pdb to mol2
        """
        wrappers.shell_call(['obabel', '-ipdb', self._input_pdb_file, '-omol2', '-O', self._input_mol2_file])

    def _prepare_for_sampling(self):
        """
        Merge receptor (if present) and peptide into a single chain and renumber
        """
        pep = BasePDB(ag=self.pep.copy())
        pep.ag.setChids('B')

        if self.rec:
            rec = BasePDB(ag=self.rec)
            rec.ag.setChids('A')
            merged = rec + pep
            for i, r in enumerate(merged.ag.iterResidues(), 1):
                r.setResnum(i)
            merged.ag.setChids('A')
            merged.ag.setSerials(range(1, merged.ag.numAtoms() + 1))
            merged = merged.ag
        else:
            merged = pep.renumber_residues(keep_resi=False).ag

        prody.writePDB(self._input_pdb_file, merged)
        self._input = merged
        self._pdb_to_mol2()

    @staticmethod
    def _scwrl_convert_atom_name(x):
        """
        Brikard flips the digits to the beginning, fix it
        """
        if not x[0].isdigit():
            return x
        else:
            x = x[1:].strip() + x[0]
            if len(x) == 4:
                return x
            else:
                return ' %-3s' % x

    def _fix_brikard_output(self):
        """
        Fix naming and numbering in brikard output
        Receptor chain becomes A (if present), peptide - B, residues and atoms
        are renumbered correspondingly
        """
        self.brikard.setNames(list(map(self._scwrl_convert_atom_name, self.brikard.getNames())))
        residues = list(self.brikard.iterResidues())
        for i, r in enumerate(residues[:-self.pep_len], 1):
            r.setResnum(i)
            r.setChids('A')
        for i, r in enumerate(residues[-self.pep_len:], 1):
            r.setResnum(i)
            r.setChids('B')
        self.brikard.setSerials(list(range(1, self.brikard.numAtoms() + 1)))

    def _find_disu_bonds(self, ag):
        """
        Find disulfide bonds by distances between SG atoms
        """
        cys_sg = ag.select('resname CYS and name SG')
        if cys_sg is None:
            return []

        crds = cys_sg.getCoords()
        dmat = np.array([[np.sqrt(((i - j) ** 2).sum()) for j in crds] for i in crds])
        pairs = zip(*map(list, np.where(dmat < self._disu_dist)))
        pairs = [p for p in pairs if p[0] < p[1]]

        resnums = cys_sg.getResnums()
        pairs = [tuple(resnums[list(p)]) for p in pairs]
        self._disu_pairs = pairs
        return pairs

    def generate_peptides(self, nsamples, nrotamers, vdw, seed, sample_resi_within=None, auto_disu=True, keep_rec=True):
        vdw_min = self._vdw_min
        vdw_max = vdw

        logger.info('Preparing structure for sampling')
        self._prepare_for_sampling()

        # identify peptide residue ids in the merged structure
        residues = list(map(int, self._input.select('name CA').getResnums()[-self.pep_len:]))
        resin = residues[0]   # N-terminus
        resic = residues[-1]  # C-terminus

        # find which receptor residues to sample
        sample_resi_list = []
        if sample_resi_within is not None:
            if self.rec is None:
                logger.warning('`sample_resi_within` is set, but receptor is missing')
            else:
                logger.info('Finding residues on the receptor to sample')

                sel = '(same residue as exwithin %f of (resnum %i:%i)) and name CA' % (sample_resi_within, resin, resic)
                sel = self._input.select(sel)

                # dont sample disulfide bonds
                if auto_disu:
                    disu_pairs = self._find_disu_bonds(self._input)
                    if disu_pairs:
                        resi_exclude = list(set(list(itertools.chain(*disu_pairs))))
                        sel = sel.select('not resnum ' + ' '.join(map(str, resi_exclude)))

                if sel:
                    sample_resi_list = list(sel.getResnums())

        restrictions = {  # TODO: add restrictions
            # (1, 2): '(-180 -150 75 180 0.2)',  # [-180., -160., 75., 180.],
            # (2, 1): '-160 -40',  # [-115., -40.],
            # (2, 2): '(-180 -160 120 180 0.2)',  # [-180., -160., 120., 180.],
            # (3, 1): '-150 -45',  # [-150., -45.],
            # (3, 2): '(-180 -140 110 180 0.3)',  # [-180., -140., 110., 180.],
            # (4, 1): '-175 -30',  # [-175., -30.],

            # (nres, 1):     '-180 -40',  # [-180, -40.],
            # (nres - 1, 2): '(-180 -140 80 180 0.4)',  # [-180, -140., 80., 180.],
            # (nres - 1, 1): '-165 -30',  # [-165, -30.],
            # (nres - 2, 1): '-175 -30'  # [-175, -30.]
        }
        restrictions = {(residues[resi - 1], tor): v for (resi, tor), v in restrictions.items()}

        for _vdw in reversed(list(np.arange(vdw_min, vdw_max + 0.0001, 0.05))):
            logger.info("Trying VDW %.3f" % _vdw)

            self._run_brikard(nsamples,
                              resin,
                              resic,
                              nrot=nrotamers,
                              vdw=_vdw,
                              seed=seed,
                              rec_resi_list=sample_resi_list,
                              restrictions=restrictions)

            if self.brikard is not None:
                logger.info('Fixing names and chains in output structures')
                self._fix_brikard_output()

                if not keep_rec:
                    logger.info('Removing receptor from the output')
                    self.brikard = self.brikard.select('chain B').copy()

                # keep only requested number of conformations
                self.brikard.setCoords(self.brikard.getCoordsets()[:nsamples])

                BasePDB(ag=self.brikard).save(self._brikard_file)

            if self.brikard is None or self.brikard.numCoordsets() < nsamples:
                logger.info("===== Current VDW of %.3f is too high" % _vdw)
            else:
                logger.info("===== Enough conformations was generated. Breaking the loop ..")
                break

        return self.brikard
