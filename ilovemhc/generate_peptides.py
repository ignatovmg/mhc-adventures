# requires:
# - BioPython
# - brikard
# - chimera (module load chimera)

from glob import glob
from Bio.SeqUtils import seq1, seq3
from Bio.Alphabet.IUPAC import IUPACProtein
import numpy as np
import subprocess
from time import time
from subprocess import Popen
import prody
from path import Path

from ilovemhc import define, wrappers
from ilovemhc.mhc_peptide import BasePDB

import logging
logger = logging.getLogger(__name__)

_chimera_script = '''open {0}
write format {1} atomTypes sybyl 0 {2}
close all
'''

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
    disu_dist = 2.5
    vdw_min = 0.0
    sampling_init_timeout = 5 * 60

    def __init__(self, pep_seq=None, pep=None, rec=None, wdir='.', prepare=True):
        self.wdir = Path(wdir)
        self.seq_file = self.wdir.joinpath('seq_file')
        self.scw_file = self.wdir.joinpath('pep_scw.pdb')
        self.mrg_file = self.wdir.joinpath('merged.pdb')
        self.input_mol2_file = self.wdir.joinpath('mol.mol2')
        self.input_pdb_file = self.wdir.joinpath('input.pdb')
        self.chim_file = self.wdir.joinpath('chimera.cmd')
        self.brikard_file = self.wdir.joinpath('brikard.pdb')
        self.a_file = self.wdir.joinpath('a.mhc')
        #self.log_file = self.wdir.joinpath('log')

        self.rec = rec
        if not rec is None and not isinstance(rec, prody.AtomGroup):
            # prepare receptor
            self.rec = prody.parsePDB(rec)
            if prepare:
                self.rec = BasePDB(ag=self.rec).prepare_pdb22('mhc').hsd_to_his().ag

        self.pep_seq = pep_seq
        if not pep_seq is None:
            self._check_sequence(pep_seq)
            # make peptide starting conformation
            self.make_pep_from_seq()
        else:
            self.pep = pep
            if not pep is None and not isinstance(pep, prody.AtomGroup):
                self.pep = prody.parsePDB(pep)

        # prepare peptide
        if prepare:
            self.pep = BasePDB(ag=self.pep).prepare_pdb22('pep').hsd_to_his().ag
        self.pep_len = self.pep.numResidues()

        self.tpl = None
        self.input = None
        self.brikard = None
        self.disu_pairs = None

    def clean(self):
        self.seq_file.remove_p()
        self.scw_file.remove_p()
        self.mrg_file.remove_p()
        self.input_mol2_file.remove_p()
        self.input_pdb_file.remove_p()
        self.chim_file.remove_p()
        self.wdir.joinpath('test.pdb').remove_p()
        self.wdir.joinpath('test0.pdb').remove_p()
        self.wdir.joinpath('test1.pdb').remove_p()
        self.wdir.joinpath('assemble.tmp').remove_p()
        if self.brikard_raw_file:
            self.brikard_raw_file.remove_p()

    def _check_sequence(self, pep_seq):
        if len(pep_seq) < 8 or len(pep_seq) > 14:
            raise RuntimeError('Peptides sequence cannot be longer than 14 or shorter than 8 residues')
        for a in pep_seq:
            if a not in IUPACProtein.letters:
                raise RuntimeError('Residue %s is not a standard amino acid' % a)

    def _generate_template(self):
        bbnames = ['C', 'O', 'CA', 'N', 'OXT']
        lgt = len(self.pep_seq)
        tpl = Path(define.PEPTIDE_TEMPLATES_DIR).joinpath('%imer.pdb' % lgt)
        tpl = prody.parsePDB(tpl)

        for r, newname in zip(tpl.iterResidues(), self.pep_seq):
            r.setResname(seq3(newname).upper())
        tpl = tpl.select('name ' + ' '.join(bbnames)).copy()
        tpl.setChids('B')
        self.tpl = tpl

    def _make_scwrl_sequence_file(self):
        l = self.tpl.select('name CA').getResnames()
        r = []
        if self.rec is not None:
            r = self.rec.select('name CA').getResnames()

        seq = ''.join(list(map(seq1, r))).lower() + ''.join(list(map(seq1, l))).upper()
        self.seq_file.write_text(seq + '\n')

    def _generate_sidechains_scwrl(self):
        if not self.rec is None:
            rec = self.rec.copy()
            rec.setChids('A')
            lig = self.tpl.copy()
            rec = BasePDB(ag=rec)
            lig = BasePDB(ag=lig)
            merged = rec.add_mol(lig, keep_resi=False, keep_chains=True)
            merged.save(self.mrg_file)

            self._make_scwrl_sequence_file()

            call = [define.SCWRL_EXE, '-h', '-i', self.mrg_file, '-o', self.scw_file, '-s', self.seq_file]
        else:
            prody.writePDB(self.mrg_file, self.tpl)
            call = [define.SCWRL_EXE, '-h', '-i', self.mrg_file, '-o', self.scw_file]

        wrappers.shell_call(call)

        pep = prody.parsePDB(self.scw_file)

        # extract peptide and renumber
        pep = BasePDB(ag=pep.select('chain B').copy()).renumber(keep_resi=False).ag
        self.pep = pep

    def make_pep_from_seq(self):
        self._generate_template()
        self._generate_sidechains_scwrl()

    def run_brikard(self, N, resin, resic, nrot=1, vdw=0.3, seed=123, rec_resi_list=None, restrictions=None):

        # pivots = (resin + 1, (resin + resic) / 2, resic - 1)
        outdir = self.wdir
        oldpwd = Path.getcwd()
        outdir.chdir()

        residues = range(resin, resic + 1)
        pivots = tuple(residues[3:6])

        sampled = set(range(resin + 1, resic))
        sampled = sorted(list(sampled - set(pivots)))
        pivots = '{0}(N) {0}(CA) {1}(N) {1}(CA) {2}(N) {2}(CA)'.format(*pivots)
        sampled = '%i(2) %s %i(1)' % (resin, " ".join([str(x) for x in sampled]), resic)

        try:
            for f in glob('mol_000001*.pdb'):
                Path(f).remove_p()

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
                for (resi, torsion), limits in sorted(restrictions.iteritems()):
                    assemble_file_content += '{:d} {:d} {}\n'.format(resi, torsion, limits)

            a_file = self.a_file.basename()
            a_file.write_text(assemble_file_content)

            # run brikard
            wrappers.shell_call([define.ASSEMBLE_EXE, a_file])
            echo = subprocess.Popen(('echo', str(seed)), stdout=subprocess.PIPE)
            process = Popen(define.BRIKARD_EXE, stdin=echo.stdout, stdout=subprocess.PIPE)

            noutputs = 0
            output = ''
            start_time = time()
            while output or (process.poll() is None):
                output = process.stdout.readline()
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
                    if time() - start_time > self.sampling_init_timeout:
                        logger.info('Couldn\'t find the first lead in 5 minutes, decrease VDW penalty')
                        process.kill()

            brikard_raw = glob('mol_000001*.pdb')
            if len(brikard_raw) == 0:
                oldpwd.chdir()
                return

            brikard_raw = brikard_raw[0]
            self.brikard_raw_file = outdir.joinpath(Path(brikard_raw))
            self.brikard = prody.parsePDB(brikard_raw)

        except:
            oldpwd.chdir()
            raise

        oldpwd.chdir()

    def _chimera_convert(self):
        cmd = self.chim_file
        with open(cmd, 'w') as f:
            f.write(_chimera_script.format(self.input_pdb_file.basename(), 'mol2', self.input_mol2_file.basename()))
        wrappers.shell_call(['chimera', '--nogui', cmd])

    def _prepare_for_sampling(self):
        pep = BasePDB(ag=self.pep.copy())
        pep.ag.setChids('B')

        if self.rec:
            rec = BasePDB(ag=self.rec)
            rec.ag.setChids('A')
            merged = rec + pep
            for i, r in enumerate(merged.ag.iterResidues(), 1):
                r.setResnum(i)
            merged.ag.setChids('A')
            merged.ag.setSerials(range(1, merged.ag.numAtoms()+1))
            merged = merged.ag
        else:
            merged = pep.renumber(keep_resi=False).ag

        prody.writePDB(self.input_pdb_file, merged)
        self.input = merged
        self._chimera_convert()

    @staticmethod
    def _scwrl_convert_atom_name(x):
        if not x[0].isdigit():
            return x
        else:
            x = x[1:].strip() + x[0]
            if len(x) == 4:
                return x
            else:
                return ' %-3s' % x

    def _fix_brikard_output(self):
        self.brikard.setNames(map(self._scwrl_convert_atom_name, self.brikard.getNames()))
        residues = list(self.brikard.iterResidues())
        for i, r in enumerate(residues[:-self.pep_len], 1):
            r.setResnum(i)
            r.setChids('A')
        for i, r in enumerate(residues[-self.pep_len:], 1):
            r.setResnum(i)
            r.setChids('B')
        self.brikard.setSerials(range(1, self.brikard.numAtoms() + 1))

    def _find_disu_bonds(self, ag):
        cys_sg = ag.select('resname CYS and name SG')
        crds = cys_sg.getCoords()
        dmat = np.array([[np.sqrt(((i - j) ** 2).sum()) for j in crds] for i in crds])
        pairs = zip(*map(list, np.where(dmat < self.disu_dist)))
        pairs = [p for p in pairs if p[0] < p[1]]

        resnums = cys_sg.getResnums()
        pairs = [tuple(resnums[list(p)]) for p in pairs]
        self.disu_pairs = pairs
        return pairs

    def generate_peptides(self, nsamples, nrotamers, vdw, seed, sample_resi_within=None, auto_disu=True, keep_rec=True):
        vdw_min = self.vdw_min
        vdw_max = vdw

        self._prepare_for_sampling()

        # identify peptide residues in the merged structure
        residues = map(int, self.input.select('name CA').getResnums()[-len(self.pep_seq):])
        resin = residues[0]  # N-terminus
        resic = residues[-1]  # C-terminus

        sample_resi_list = []
        if sample_resi_within is not None:
            sel = '(same residue as exwithin %f of (resnum %i:%i)) and name CA' % (sample_resi_within, resin, resic)
            sel = self.input.select(sel)

            if auto_disu:
                disu_pairs = self._find_disu_bonds(self.input)
                resi_exclude = list(set(reduce(lambda x, y: list(x) + list(y), disu_pairs)))
                sel = sel.select('not resnum ' + ' '.join(map(str, resi_exclude)))

            if sel:
                sample_resi_list = list(sel.getResnums())

        restrictions = {
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
        restrictions = {(residues[resi-1], tor): v for (resi, tor), v in restrictions.iteritems()}
        #restrictions = None

        for _vdw in reversed(list(np.arange(vdw_min, vdw_max + 0.0001, 0.05))):
            logger.info("Trying VDW %.3f" % _vdw)

            self.run_brikard(nsamples,
                             resin,
                             resic,
                             nrot=nrotamers,
                             vdw=_vdw,
                             seed=seed,
                             rec_resi_list=sample_resi_list,
                             restrictions=restrictions)

            if self.brikard is not None:
                self._fix_brikard_output()

                if not keep_rec:
                    self.brikard = self.brikard.select('chain B').copy()

                BasePDB(ag=self.brikard).save(self.brikard_file)

            if self.brikard is None or self.brikard.numCoordsets() < nsamples:
                logger.info("===== Current VDW of %.3f is too high" % _vdw)
            else:
                logger.info("===== Enough conformations was generated. Breaking the loop ..")
                break
