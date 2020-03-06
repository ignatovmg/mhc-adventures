

from ilovemhc import utils
from ilovemhc.brikard.generate_peptides import chimera_convert, run_brikard
from ilovemhc.brikard.minimization import minimize_energy_single_files_generic
import logging
import click
import prody
from path import Path


def get_loop_residues(pdb, selection):
    pdb = prody.parsePDB(pdb)
    pdb = pdb.select(selection)
    return [r.getResindex() + 1 for r in pdb.getHierView().iterResidues()]


@click.command()
@click.argument('pdb', type=click.Path(exists=True))
@click.argument('nsamples', type=int)
@click.argument('loop_selection')
@click.option('--vdw', default=0.2)
@click.option('--radius', default=4.0)
@click.option('--min_add_selection', default='')
def cli(pdb, nsamples, loop_selection, vdw, radius, min_add_selection):
    for k, v in locals().iteritems():
        logging.info("{:20s} = {}".format(str(k), str(v)))

    loop = loop_selection  # 'chain B and resnum 97 to 107'
    pdb = Path(pdb)  # 'dom4_nb694.pdb'
    nmin_orig, _ = utils.prepare_pdb22(pdb, pdb[:-4])
    nmin = utils.renumber_pdb(nmin_orig[:-4] + '_renum.pdb', nmin_orig, False, False)
    nmin = utils.hsd2his(nmin, nmin[:-4] + '_hsd.pdb')
    chimera_convert(nmin, 'mol.mol2')

    loop_rlist = sorted(get_loop_residues(nmin_orig, loop))
    nter, cter = loop_rlist[0], loop_rlist[-1]
    brikarded, nbrikarded = run_brikard(nsamples, nter, cter, vdw=vdw)

    brikarded = utils.his2hsd(brikarded)
    brikarded, _ = utils.match_by_residue_position(brikarded, nmin_orig)

    # fix CA within 4.0 of the loop and the rest of the protein
    if min_add_selection:
        selection = 'not (((not calpha) and within {rad:.2f} of ({loop})) or ({loop}) or {add})'
    else:
        selection = 'not (((not calpha) and within {rad:.2f} of ({loop})) or ({loop}))'
    selection = selection.format(loop=loop, rad=radius, add=min_add_selection)
    minimize_energy_single_files_generic(brikarded, nsteps=10000, fix_selection=selection)


if __name__ == '__main__':
    fmt = '%(asctime)s [%(levelname)s] %(message)s'
    logging.basicConfig(format=fmt, level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
    cli()
