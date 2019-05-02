# requires:
# - BioPython
# - brikard
# - chimera (module load chimera)

import sys
import os
from glob import glob
import pandas as pd

from Bio.SeqUtils import seq1, seq3
from Bio.Alphabet.IUPAC import IUPACProtein
import numpy as np
from scipy.spatial.distance import cdist
import subprocess
from time import time, sleep
from subprocess import Popen, check_output
import prody

from path import Path
import click

from ilovemhc import define
from ilovemhc import utils, wrappers
import minimization
import pymol
from pymol import cmd

import logging
import logging.config
#logging.config.fileConfig(define.LOGGING_CONF)
#logger = logging.getlogging('root')


def check_file(path):
    path = Path(path)
    return not path.isfile() or path.size == 0


def file_error(path):
    if check_file(path):
        raise OSError("%s doesn't exist or is empty" % path)

        
def generate_template(seq, save_path):
    newpep = save_path
    lgt = len(seq)
    tpl = Path(define.PEPTIDE_TEMPLATES_DIR).joinpath('%imer.pdb' % lgt)
    file_error(tpl)

    anames = ['C', 'O', 'CA', 'N', 'OXT']
    with open(tpl, 'r') as f, open(newpep, 'w') as g:
        i = 0
        for line in f:
            if line.startswith('END'):
                break
            elif line.startswith('ATOM'):
                aname = line[utils.pdb_slices['atomn']]
                if aname.strip() not in anames:
                    continue
                resn = line[utils.pdb_slices['resn']]
                resi = int(line[utils.pdb_slices['resi']])
                newresn = seq3(seq[resi-1]).upper()
                newline = utils.change_atom_record(line, resn=newresn)
                g.write(newline)
                i += 1
        g.write('END')
        
    file_error(save_path)


def make_scwrl_sequence_file(rec, lig, out):
    rec = Path(rec)
    file_error(rec)

    lig = Path(lig)
    file_error(lig)

    #outdir = out.dirname()
    with open(rec, 'r') as r, open(lig, 'r') as l:
        rlines = filter(lambda x: x.startswith('ATOM'), r)
        llines = filter(lambda x: x.startswith('ATOM'), l)
        rlist = [(int(x[utils.pdb_slices['resi']]), seq1(x[utils.pdb_slices['resn']])) for x in rlines]
        rlist = sorted(list(set(rlist)))
        llist = [(int(x[utils.pdb_slices['resi']]), seq1(x[utils.pdb_slices['resn']])) for x in llines]
        llist = sorted(list(set(llist)))
        resn_list = map(str.lower, zip(*rlist)[1]) + map(str.upper, zip(*llist)[1])

    #out = outdir.joinpath('seq_file.txt')
    out.write_text(''.join(resn_list) + '\n')


def generate_sidechains_scwrl(pdb, save_path, rec=None):
    file_error(pdb)

    save_path = Path(save_path)
    outdir = save_path.dirname()

    if rec:
        merged_tmp = outdir.joinpath('merged_tmp.pdb')
        utils.merge_two(merged_tmp, rec, pdb, keep_residue_numbers=True, keep_chain=True)

        sequence_file = outdir.joinpath('sequence_file')
        make_scwrl_sequence_file(rec, pdb, sequence_file)

        call = [define.SCWRL_EXE, '-h', '-i', merged_tmp, '-o', save_path, '-s', sequence_file]
    else:
        call = [define.SCWRL_EXE, '-h', '-i', pdb, '-o', save_path]

    #try:
    #    output = check_output(call)
    #except Exception as e:
    #    logging.error(e.output)
    #    raise RuntimeError('SCWRL failed')
    wrappers.shell_call(call)

    if rec:
        merged_tmp.remove()
        with open(save_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith('TER'):
                    break
            lines = lines[i+1:]
            lines = filter(lambda x: x.startswith('ATOM'), lines) + ['END\n']
            utils.renumber_pdb(save_path, lines)

    #logging.info(output)
    file_error(save_path)


def preminimize_peptide(mhc, peptide, out):
    mhc = Path(mhc)
    peptide = Path(peptide)
    out = Path(out)

    outdir = out.dirname()
    pmhc_nonmin = outdir.joinpath('pmhc_nonmin.pdb')
    utils.merge_two(pmhc_nonmin, mhc, peptide, keep_chain=True, keep_residue_numbers=True)
    with open(pmhc_nonmin, 'r') as f:
        lines = filter(lambda x: x.startswith('ATOM'), map(str.strip, f.readlines()))
    pmhc_nonmin.write_lines(['MODEL 1'] + lines + ['ENDMDL'])
    pmhc_min = outdir.joinpath('pmhc_min.pdb')
    _, nmin_tmp, psf_tmp = minimization.minimize_energy(pmhc_nonmin, out=pmhc_min, nsteps=100, fix_radius=0.0)

    Path(nmin_tmp).remove()
    Path(psf_tmp).remove()

    with open(pmhc_min, 'r') as f, open(out, 'w') as o:
        for line in f:
            if line.startswith('ATOM') and line[utils.pdb_slices['chain']] == 'B':
                o.write(line)

    pmhc_nonmin.remove()
    pmhc_min.remove()
    utils.hsd2his(out)


def find_resi_within(rec, lig, radius=4.0):
    with open(rec, 'r') as r, open(lig, 'r') as l:
        resn_s = utils.pdb_slices['resn']  # to keep DISU bonds unperturbed
        rec_lines = filter(lambda x: x.startswith('ATOM') and x[resn_s] != 'CYS', r)
        lig_lines = filter(lambda x: x.startswith('ATOM'), l)
    x, y, z = utils.pdb_slices['x'], utils.pdb_slices['y'], utils.pdb_slices['z']
    rcrd = np.array([map(float, [line[x], line[y], line[z]]) for line in rec_lines])
    lcrd = np.array([map(float, [line[x], line[y], line[z]]) for line in lig_lines])
    dist = cdist(rcrd, lcrd)
    mask = (dist < radius).any(axis=1)
    contacts = np.array(rec_lines)[mask]
    resi_list = list(set([int(line[utils.pdb_slices['resi']]) for line in contacts]))

    return resi_list


_chimera_script = '''open {0}
write format {1} atomTypes sybyl 0 {2}
close all
'''


def chimera_convert(pdb, mol2):
    pdb = Path(pdb)
    mol2 = Path(mol2)
    file_error(pdb)
    
    cmd = mol2.dirname().joinpath('chimera.cmd')
    with open(cmd, 'w') as f:
        f.write(_chimera_script.format(pdb.basename(), 'mol2', mol2.basename()))

    utils.shell_call(['chimera', '--nogui', cmd])
    file_error(mol2)

    
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


def run_brikard(N, resin, resic, outdir=Path('.'), nrot=1, vdw=0.3, seed=123, rec_resi_list=None, restrictions=None):

    # pivots = (resin + 1, (resin + resic) / 2, resic - 1)
    residues = range(resin, resic + 1)
    pivots = tuple(residues[3:6])

    sampled = set(range(resin+1, resic))
    sampled = sorted(list(sampled - set(pivots)))
    pivots = '{0}(N) {0}(CA) {1}(N) {1}(CA) {2}(N) {2}(CA)'.format(*pivots)
    sampled = '%i(2) %s %i(1)' % (resin, " ".join([str(x) for x in sampled]), resic)

    oldpwd = Path.getcwd()
    outdir.chdir()
    try:
        outdir = Path('.')
        assemble_file_content = _assemble_production.format(brikard_lib=define.BRIKARD_LIB,
                                                            output_dir=outdir,
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

        a_file = outdir.joinpath('a.mhc')
        with open(a_file, 'w') as f:
            f.write(assemble_file_content)

        # run brikard
        utils.shell_call([define.ASSEMBLE_EXE, a_file])
        echo = subprocess.Popen(('echo', str(seed)), stdout=subprocess.PIPE)
        process = Popen(define.BRIKARD_EXE, stdin=echo.stdout, stdout=subprocess.PIPE)

        noutputs = 0
        output = ''
        start_time = time()
        while output or (process.poll() is None):
            output = process.stdout.readline()
            if output:
                logging.info(output.strip())
                if 'accepted =' in output:
                    naccepted = int(output.split('accepted = ')[-1])
                    noutputs += 1
                    progress = naccepted / float(noutputs)

                    # Kill if progress is too slow
                    if noutputs > 50 and progress < 0.05:
                        logging.info("Progress is too slow: %f. Killing ..." % progress)
                        process.kill()
                    if naccepted > N:
                        logging.info("Brikard has generated %i (more than requested).." % naccepted)
                        #process.kill()

            # Kill if cannot initialize sampling in 5 mins
            if noutputs == 0:
                if time() - start_time > 5 * 60:
                    logging.info('Couldn\'t find the first lead in 5 minutes, decrease VDW penalty')
                    process.kill()

        brikard_raw = outdir.joinpath('mol_000001.pdb')
        brikard_out = outdir.joinpath('brikard.pdb')
        resi_slice = utils.pdb_slices['resi']
        atomn_slice = utils.pdb_slices['atomn']

        nbrikarded = 0
        new_resi = 0
        prv_resi = -1

        def scwrl_convert_atom_name(x):
            if not x[0].isdigit():
                return x
            else:
                x = x[1:].strip() + x[0]
                if len(x) == 4:
                    return x
                else:
                    return ' %-3s' % x

        if brikard_raw.exists():
            with open(brikard_raw, 'r') as f, open(brikard_out, 'w') as o:
                for linei, line in enumerate(f):
                    if line.startswith('MODEL'):
                        new_resi = 0
                        prv_resi = -1

                    if line.startswith('ATOM'):
                        resi = int(line[resi_slice])
                        atomn = line[atomn_slice]
                        atomn = scwrl_convert_atom_name(atomn).strip()

                        if resi != prv_resi:
                            new_resi += 1
                            prv_resi = resi
                        if resi < resin or resi > resic:
                            line = utils.change_atom_record(line, atomn=atomn, chain='A', resi=new_resi)
                        else:
                            line = utils.change_atom_record(line, atomn=atomn, chain='B', resi=(new_resi - resin + 1))
                    if line.startswith('TER'):
                        continue

                    o.write(line)

                    if line.startswith('END'):
                        nbrikarded += 1
                        if nbrikarded == N:
                            break
            brikard_raw.remove()
    except:
        oldpwd.chdir()
        raise

    oldpwd.chdir()

    return brikard_out, nbrikarded


@click.command()
@click.argument('mhc', type=click.Path(exists=True))
@click.argument('seq')
@click.argument('nsamples', type=int)
@click.option('--nrotamers', default=1, help='Number of rotamers to sample')
@click.option('--vdw', default=0.2)
@click.option('--sample_resi_within', default=None, type=float)
@click.option('--outdir', default='.', type=click.Path(exists=True))
@click.option('--seed', default=123, help='Random seed')
def cli(*args, **kwargs):
    generate_peptides(*args, **kwargs)


def generate_peptides(mhc, seq, nsamples, nrotamers, vdw, outdir, seed, sample_resi_within=None):
    mhc = Path(mhc)
    vdw_min = 0.05
    vdw_max = vdw
    outdir = Path(outdir)

    for k, v in locals().iteritems():
        logging.info("{:20s} = {}".format(str(k), str(v)))

    if not mhc.endswith('.pdb'):
        raise RuntimeError('File must have .pdb extension')

    if len(seq) < 8 or len(seq) > 14:
        raise RuntimeError('Peptides sequence cannot be longer than 14 or shorter than 8 residues')

    for a in seq:
        if a not in IUPACProtein.letters:
            raise RuntimeError('Residue %s is not a standard amino acid' % a)

    # fix mhc
    utils.renumber_pdb(mhc, mhc, keep_resi=False)
    mhc, _ = utils.prepare_pdb22(mhc, outdir.joinpath('mhc'))
    utils.hsd2his(mhc)

    peptide_template = outdir.joinpath('peptide.pdb')
    generate_template(seq, peptide_template)

    peptide_scwrled = outdir.joinpath('peptide_scwrl.pdb')
    generate_sidechains_scwrl(peptide_template, peptide_scwrled, rec=mhc)

    peptide_ready = outdir.joinpath('peptide_ready')
    # cant use reduce here because it didnt add 2 H's to 2mha_SRDHSRTPM once
    # utils.reduce_pdb(peptide_scwrled, peptide_ready, trim_first=True)
    peptide_ready, _ = utils.prepare_pdb22(peptide_scwrled, peptide_ready)

    peptide_min = outdir.joinpath('peptide_min.pdb')
    preminimize_peptide(mhc, peptide_ready, peptide_min)

    peptide_final = peptide_min

    sample_resi_list = []
    if sample_resi_within is not None:
        sample_resi_list = find_resi_within(mhc, peptide_final, sample_resi_within)

    merged_pdb = outdir.joinpath('merged.pdb')
    utils.merge_two(merged_pdb, mhc, peptide_final)

    # identify peptide residues in the merged structure
    with open(merged_pdb, 'r') as f:
        residues = filter(lambda x: x.startswith('ATOM'), f)
        residues = map(lambda x: int(x[22:26]), residues)
        residues = sorted(list(set(residues)))[-len(seq):]

    resin = residues[0]  # N-terminus
    resic = residues[-1]  # C-terminus
    nres = len(residues)
    
    # make mol2
    mol2_file = outdir.joinpath('mol.mol2')
    chimera_convert(merged_pdb, mol2_file)

    restrictions = {
        #(1, 2): '(-180 -150 75 180 0.2)',  # [-180., -160., 75., 180.],
        #(2, 1): '-160 -40',  # [-115., -40.],
        #(2, 2): '(-180 -160 120 180 0.2)',  # [-180., -160., 120., 180.],
        #(3, 1): '-150 -45',  # [-150., -45.],
        #(3, 2): '(-180 -140 110 180 0.3)',  # [-180., -140., 110., 180.],
        #(4, 1): '-175 -30',  # [-175., -30.],

        #(nres, 1):     '-180 -40',  # [-180, -40.],
        #(nres - 1, 2): '(-180 -140 80 180 0.4)',  # [-180, -140., 80., 180.],
        #(nres - 1, 1): '-165 -30',  # [-165, -30.],
        #(nres - 2, 1): '-175 -30'  # [-175, -30.]
    }

    #restrictions = {(residues[resi-1], tor): v for (resi, tor), v in restrictions.iteritems()}
    restrictions = None

    for _vdw in reversed(list(np.arange(vdw_min, vdw_max + 0.0001, 0.05))):
        logging.info("Trying VDW %.3f" % _vdw)

        out_pdb, nbrikarded = run_brikard(nsamples, 
                                          resin, 
                                          resic,
                                          outdir=outdir, 
                                          nrot=nrotamers, 
                                          vdw=_vdw, 
                                          seed=seed,
                                          rec_resi_list=sample_resi_list,
                                          restrictions=restrictions)

        if nbrikarded < nsamples:
            logging.info("===== Current VDW of %.3f is too high" % _vdw)
        else:
            logging.info("===== Enough conformations was generated. Breaking the loop ..")
            break

            
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
    cli()
