import generate_peptides
import minimization
import atom_naming
import rosetta
from ilovemhc import utils, define
from path import Path
import pandas as pd

import logging
import click


@click.command()
@click.argument('mhc', type=click.Path(exists=True))
@click.argument('pseq')
@click.argument('nsamples', type=int)
@click.option('--outdir', default='.', type=click.Path())
@click.option('--minimize', is_flag=True)
@click.option('--rosetta_score', is_flag=True)
@click.option('--ref_pdb', default=None, type=click.Path())
@click.option('--vdw', default=0.3)
@click.option('--sample_resi_within', default=0.0)
def cli(*args, **kwargs):
    pipeline(*args, **kwargs)


def pipeline(mhc, pseq, nsamples,
             outdir='.',
             minimize=True,
             rosetta_score=True,
             ref_pdb=None,
             vdw=0.2,
             sample_resi_within=0.0,
             minimize_resi_within=4.0):

    mhc = Path(mhc)
    logging.info(mhc)

    if ref_pdb:
        ref_pdb = Path(ref_pdb)
        assert(ref_pdb.exists())

    outdir = Path(outdir)
    if not outdir.exists():
        outdir.mkdir_p()

    mhccpy = outdir.joinpath('mhc.pdb')
    mhc.copyfile(mhccpy)
    mhc = mhccpy

    refcpy = outdir.joinpath('ref.pdb')
    ref_pdb.copyfile(refcpy)
    ref_pdb = refcpy

    logging.info('Peptide generation')
    generate_peptides.generate_peptides(mhc, pseq, nsamples,
                                        nrotamers=1,
                                        vdw=vdw,
                                        outdir=outdir,
                                        seed=123456,
                                        sample_resi_within=sample_resi_within)

    brikarded = outdir.joinpath('brikard.pdb')
    assert(brikarded.exists())

    if minimize:
        logging.info('Minimization')

        logging.info('Libmol2 minimization')
        #minimized, nonminimized, psf = minimization.minimize_energy_single_files(brikarded, fix_radius=minimize_resi_within)
        #minimized = brikarded.dirname().joinpath('minimized.pdb')
        #nonminimized = brikarded.dirname().joinpath('brikard_nmin.pdb')
        #assert (minimized.exists() and nonminimized.exists())

        rosetta_pdb = None
        if rosetta_score:
            logging.info('Rosetta minimization')
            nonminimized, psf, first_model = minimization.prepare_models(brikarded, define.PRM22_FILE, define.RTF22_FILE)
            first_model.remove()
            rosetta_csv, rosetta_pdb = rosetta.score_models(nonminimized, outdir.joinpath('rosetta.csv'), minimize=True, constraints=True)
        else:
            logging.info('Libmol2 minimization')
            minimized, nonminimized, psf = minimization.minimize_energy_single_files(brikarded, fix_radius=minimize_resi_within)
            #minimized = brikarded.dirname().joinpath('minimized.pdb')
            #nonminimized = brikarded.dirname().joinpath('brikard_nmin.pdb')
            assert(minimized.exists() and nonminimized.exists())

        if ref_pdb:
            logging.info('RMSD calculation')
            ref_pdb, ref_psf = utils.prepare_pdb22(ref_pdb, ref_pdb[:-4])
            nmin_aa = utils.rmsd_ref_vs_models(ref_pdb, nonminimized, backbone=False)
            nmin_bb = utils.rmsd_ref_vs_models(ref_pdb, nonminimized, backbone=True)
            min_aa = utils.rmsd_ref_vs_models(ref_pdb, minimized, backbone=False)
            min_bb = utils.rmsd_ref_vs_models(ref_pdb, minimized, backbone=True)
            rmsd = [nmin_aa, nmin_bb, min_aa, min_bb]

            if rosetta_pdb:
                ros_aa = utils.rmsd_ref_vs_models(ref_pdb, rosetta_pdb, backbone=False)
                ros_bb = utils.rmsd_ref_vs_models(ref_pdb, rosetta_pdb, backbone=True)
                rmsd += [ros_aa, ros_bb]

            # convert to Series in order to match model indices in case if some models were skipped
            rmsd = [zip(*x) for x in rmsd]
            rmsd = [pd.Series(x[1], index=x[0]) for x in rmsd]
            rmsd = pd.DataFrame(rmsd).transpose()
            cols = ['nmin_aa', 'nmin_bb', 'min_aa', 'min_bb']

            if rosetta_score:
                cols += ['ros_aa', 'ros_bb']

            rmsd.columns = cols
            rmsd.to_csv(outdir.joinpath('rmsd.csv'), float_format='%.3f')
            #if rosetta_score:
            #    rmsd.to_csv(outdir.joinpath('rmsd_rosetta.csv'), float_format='%.3f')
            #else:
            #    rmsd.to_csv(outdir.joinpath('rmsd_libmol.csv'), float_format='%.3f')
    logging.info('Finished pipeline')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.DEBUG,
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    cli()
