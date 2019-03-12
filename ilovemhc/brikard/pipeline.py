import generate_peptides
import minimization
import rosetta
from ilovemhc import utils
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
def cli(*args, **kwargs):
    pipeline(*args, **kwargs)


def pipeline(mhc, pseq, nsamples, outdir='.', minimize=True, rosetta_score=True, ref_pdb=None):
    mhc = Path(mhc)

    if ref_pdb:
        ref_pdb = Path(ref_pdb)
        assert(ref_pdb.exists())

    outdir = Path(outdir)
    if not outdir.exists():
        outdir.mkdir_p()

    mhccpy = outdir.joinpath('mhc.pdb')
    mhc.copyfile(mhccpy)
    mhc = mhccpy
    utils.renumber_pdb(mhc, mhc, keep_resi=False)
    mhc, _ = utils.prepare_pdb22(mhc, outdir.joinpath('mhc'))
    utils.hsd2his(mhc)

    #refcpy = outdir.joinpath('ref.pdb')
    ##ref_pdb.copyfile(refcpy)
    #ref_pdb = refcpy
    if ref_pdb:
        ref_pdb, _ = utils.prepare_pdb22(ref_pdb, outdir.joinpath('ref'))
        utils.hsd2his(ref_pdb)
        utils.renumber_pdb(ref_pdb, ref_pdb, keep_resi=False)

    logging.info('Peptide generation')

    generate_peptides.generate_peptides(mhc, pseq, nsamples,
                                        nrotamers=1,
                                        vdw=0.2,
                                        outdir=outdir,
                                        seed=123456,
                                        sample_resi_within=3.0)

    brikarded = outdir.joinpath('brikard.pdb')
    assert(brikarded.exists())

    if minimize:
        logging.info('Minimization')
        minimized, nonminimized, psf = minimization.minimize_energy(brikarded)
        assert(minimized.exists() and nonminimized.exists())

        if rosetta_score:
            logging.info('Rosetta scoring')
            rosetta.score_models(minimized, outdir.joinpath('minimized-rosetta.csv'))

        if ref_pdb:
            logging.info('RMSD calculation')
            ref_pdb, ref_psf = utils.prepare_pdb22(ref_pdb, ref_pdb[:-4])
            nmin_aa = utils.rmsd_ref_vs_models(ref_pdb, nonminimized, backbone=False)
            nmin_bb = utils.rmsd_ref_vs_models(ref_pdb, nonminimized, backbone=True)
            min_aa = utils.rmsd_ref_vs_models(ref_pdb, minimized, backbone=False)
            min_bb = utils.rmsd_ref_vs_models(ref_pdb, minimized, backbone=True)

            # convert to Series in order to match model indices in case if some models were skipped
            rmsd = [nmin_aa, nmin_bb, min_aa, min_bb]
            rmsd = [zip(*x) for x in rmsd]
            rmsd = [pd.Series(x[1], index=x[0]) for x in rmsd]
            rmsd = pd.DataFrame(rmsd).transpose()
            rmsd.columns = ['nmin_aa', 'nmin_bb', 'min_aa', 'min_bb']
            rmsd.to_csv(outdir.joinpath('rmsd.csv'), float_format='%.3f')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.DEBUG,
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    cli()
