from ilovemhc import utils, define, wrappers
from path import Path
import numpy as np

import click
import logging
import subprocess
import prody


def prepare_models(pdb, prm, rtf, **kwargs):
    pdb = Path(pdb)
    outdir = pdb.dirname()

    pdb_hsd = Path(pdb[:-4] + '_hsd.pdb')
    utils.his2hsd(pdb, pdb_hsd)
    first_model = Path(pdb_hsd[:-4] + '_model1.pdb')

    with open(pdb, 'r') as f, open(first_model, 'w') as o:
        for line in f:
            o.write(line)
            if line.startswith('END'):
                break

    first_model_nmin, first_model_psf = utils.prepare_pdb22(first_model,
                                                            first_model[:-4],
                                                            prm=prm,
                                                            rtf=rtf,
                                                            **kwargs)

    matched = Path(pdb[:-4] + '_nmin.pdb')
    utils.match_pdb(pdb_hsd, first_model_nmin, matched)
    pdb_hsd.remove()

    matched_psf = Path(pdb[:-4] + '_nmin.psf')
    Path(first_model_psf).rename(matched_psf)
    first_model.remove()

    return matched, matched_psf, first_model_nmin


def fix_atoms(pdb, out=None, lig_chain='B', rec_chain='A', radius=4.0):
    pdb = Path(pdb)
    if out is None:
        out = Path(pdb.stripext() + '-fixed.pdb')

    if radius > 0.0:
        p = prody.parsePDB(pdb)
        sel = p.select('(chain {rec} exwithin {rad:.1f} of chain {lig}) or \
        (backbone and chain {rec})'.format(rec=rec_chain, lig=lig_chain, rad=radius))
        atomi_list = sel.getSerials()

        with open(pdb, 'r') as f, open(out, 'w') as o:
            for line in f:
                if line.startswith('ATOM') and int(line[utils.pdb_slices['atomi']]) in atomi_list:
                    o.write(line)
    else:
        out.write_text('')

    return out


def fix_atoms_generic(pdb, out=None, selection=''):
    pdb = Path(pdb)
    if out is None:
        out = Path(pdb.stripext() + '-fixed.pdb')

    if selection:
        p = prody.parsePDB(pdb)
        sel = p.select(selection)
        atomi_list = sel.getSerials()

        with open(pdb, 'r') as f, open(out, 'w') as o:
            for line in f:
                if line.startswith('ATOM') and int(line[utils.pdb_slices['atomi']]) in atomi_list:
                    o.write(line)
    else:
        out.write_text('')

    return out


def make_springs(pdb, out=None, chain='B', weight=10.0,
                 nter_crd=(8.770, 10.594, 9.683),
                 cter_crd=(31.514, 10.064, 10.259)):
    with open(pdb, 'r') as f:
        counter = 1
        nter = None
        cter = None
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                if line[utils.pdb_slices['chain']] == chain and line[utils.pdb_slices['atomn']] == ' CA ':
                    if nter is None:
                        nter = counter

                    cter = counter
                counter += 1
            if line.startswith('END'):
                break

    pdb = Path(pdb)
    if out is None:
        out = Path(pdb.stripext() + '-springs')

    with open(out, 'w') as f:
        f.write('2\n')
        f.write('1 {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(weight, *nter_crd))
        f.write('{} CA\n'.format(nter))
        f.write('1 {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(weight, *cter_crd))
        f.write('{} CA\n'.format(cter))

    return out


def find_minization_bugs(lines, model_id=0):
    errors = []
    for line in lines:
        if line.startswith('REMARK FINAL Total:'):
            energy = float(line.split()[-1])
            if energy > 10000.0:
                err = 'Model %i energy is too large (%.3f), dropping it' % (model_id, energy)
                errors.append(err)

        if line.startswith('ATOM'):
            if utils.check_atom_record(line) is None:
                err = 'Model %i has a misconfigured ATOM record, dropping it:\n%s' % (model_id, line)
                errors.append(err)
    return errors


def fix_buggy_models(pdb, fixed=None):
    pdb = Path(pdb)
    if fixed:
        fixed = Path(fixed)
    else:
        fixed = pdb

    with open(pdb, 'r') as f:
        out_lines = f.readlines()

    slices = np.array(utils.pdb_to_slices(pdb))
    good_models = []
    buggy_models = []

    for slice_id, m in enumerate(slices):
        model = out_lines[m]
        model_id = slice_id + 1
        errors = find_minization_bugs(model, model_id)

        if errors:
            logging.warning('MODEL %i' % model_id)
            for err in errors:
                logging.warning(err)
            buggy_models.append(slice_id)
        else:
            good_models.append(slice_id)

    with open(fixed, 'w') as f:
        for s in slices[good_models]:
            f.write(''.join(out_lines[s]))

    err_out = Path(fixed.stripext() + '-errors.pdb')
    if buggy_models:
        with open(err_out, 'w') as f:
            for s in slices[buggy_models]:
                f.write(''.join(out_lines[s]))

    return fixed, err_out


@click.command()
@click.argument('pdb')
@click.option('--out', default=None, type=click.Path())
@click.option('--prm', default=define.PRM22_FILE, type=click.Path(exists=True))
@click.option('--rtf', default=define.RTF22_FILE, type=click.Path(exists=True))
@click.option('--nsteps', default=1000)
def cli(*args, **kwargs):
    minimize_energy_single_files(*args, **kwargs)


def minimize_energy(pdb,
                    out=None,
                    prm=define.PRM22_FILE,
                    rtf=define.RTF22_FILE,
                    nsteps=1000,
                    fix_radius=4.0,
                    psf=None,
                    clean=False):

    for k, v in locals().iteritems():
        logging.info("{:10s} = {}".format(str(k), str(v)))

    pdb = Path(pdb)
    logging.info('Preparing models')
    if psf is None:
        nmin_models, psf_models, nmin_first = prepare_models(pdb, prm, rtf)
    else:
        nmin_models = pdb
        psf_models = psf
        nmin_first = nmin_models

    logging.info('Writing fixed atoms')
    fixed = fix_atoms(nmin_first, radius=fix_radius)

    logging.info('Writing springs')
    springs = make_springs(nmin_first)

    logging.info('Writing protocol')
    protocol = Path(pdb.stripext() + '-protocol')
    protocol.write_text('{nsteps} {fixed} . {springs}\n'.format(nsteps=nsteps, fixed=fixed, springs=springs))

    if out is None:
        out = pdb.dirname().joinpath('minimized.pdb')

    logging.info('Running minimization')
    call = [define.MINIMIZE_EXE,
            '--pdb', nmin_models,
            '--psf', psf_models,
            '--rtf', rtf,
            '--prm', prm,
            '--out', out,
            '--protocol', protocol]

    wrappers.shell_call(call, stderr=subprocess.STDOUT)
    fix_buggy_models(out)

    if clean:
        logging.info('Cleaning')
        utils.remove_files([springs, protocol, fixed])

    return out, nmin_models, psf_models


def minimize_energy_single_files(pdb, out=None,
                                 prm=define.PRM22_FILE,
                                 rtf=define.RTF22_FILE,
                                 nsteps=1000,
                                 fix_radius=4.0,
                                 psf=None,
                                 prepare_kwargs={},
                                 clean=False):

    for k, v in locals().iteritems():
        logging.info("{:10s} = {}".format(str(k), str(v)))

    pdb = Path(pdb)
    if out is None:
        out = pdb.dirname().joinpath('minimized.pdb')
    outdir = out.dirname()

    logging.info('Preparing models')
    if psf is None:
        nmin_models, psf, first_nmin = prepare_models(pdb, prm, rtf, **prepare_kwargs)
    else:
        nmin_models = pdb
        psf = psf
        first_nmin = nmin_models

    logging.info('Writing fixed atoms')
    fixed = fix_atoms(first_nmin, radius=fix_radius)

    logging.info('Writing springs')
    springs = make_springs(first_nmin)

    logging.info('Writing protocol')
    protocol = outdir.joinpath('protocol')
    protocol.write_text('{nsteps} {fixed} . {springs}\n'.format(nsteps=nsteps, fixed=fixed, springs=springs))

    nmin_list = utils.split_models(nmin_models, outdir)
    minimized_list = []
    errors_list = []
    #out_stream = open(out, 'w')

    for modeli, nmin in enumerate(nmin_list, 1):
        logging.info('Minimizing %s' % nmin)
        logging.info('Running minimization')
        out_local = outdir.joinpath(nmin.basename()[:-4] + '_min.pdb')
        call = [define.MINIMIZE_EXE,
                '--pdb', nmin,
                '--psf', psf,
                '--rtf', rtf,
                '--prm', prm,
                '--out', out_local,
                '--protocol', protocol]

        try:
            wrappers.shell_call(call, stderr=subprocess.STDOUT)
        except Exception as e:
            logging.exception(e)
            errors_list.append((out_local, [e.output]))
            continue

        if out_local.exists():
            with open(out_local, 'r') as f:
                errors = find_minization_bugs(f.readlines())
        else:
            errors = [out_local + ' doesn\'t exist']

        if errors:
            for e in errors:
                logging.error(e)
            errors_list.append((out_local, errors))
        else:
            minimized_list.append(out_local)
            #with open(out_local, 'r') as f:
            #    out_stream()
            nmin.remove()

    if errors_list:
        err_file = outdir.joinpath('min_errors.log')
        with open(err_file, 'w') as f:
            for err_pdb, errors in errors_list:
                f.write('\n'.join(['\n', err_pdb] + errors))

    utils.assemble_models(minimized_list, out, remove=True)

    if clean:
        logging.info('Cleaning')
        utils.remove_files([springs, protocol, fixed])

    return out, nmin_models, psf


def minimize_energy_single_files_generic(pdb, out=None,
                                         prm=define.PRM22_FILE,
                                         rtf=define.RTF22_FILE,
                                         nsteps=1000,
                                         fix_selection='',
                                         prepare_kwargs={}):
    for k, v in locals().iteritems():
        logging.info("{:10s} = {}".format(str(k), str(v)))

    pdb = Path(pdb)
    if out is None:
        out = pdb.dirname().joinpath('minimized.pdb')
    outdir = out.dirname()

    logging.info('Preparing models')
    nmin_models, psf, first_nmin = prepare_models(pdb, prm, rtf, **prepare_kwargs)

    nmin_list = utils.split_models(nmin_models, outdir)
    minimized_list = []
    errors_list = []
    for modeli, nmin in enumerate(nmin_list, 1):
        logging.info('Writing fixed atoms')
        fixed = fix_atoms_generic(nmin, selection=fix_selection)

        logging.info('Writing protocol')
        protocol = outdir.joinpath('protocol')
        protocol.write_text('{nsteps} {fixed} . {springs}\n'.format(nsteps=nsteps, fixed=fixed, springs='.'))

        logging.info('Minimizing %s' % nmin)
        logging.info('Running minimization')
        out_local = outdir.joinpath(nmin.basename()[:-4] + '_min.pdb')
        call = [define.MINIMIZE_EXE,
                '--pdb', nmin,
                '--psf', psf,
                '--rtf', rtf,
                '--prm', prm,
                '--out', out_local,
                '--protocol', protocol]

        try:
            wrappers.shell_call(call, stderr=subprocess.STDOUT)
        except Exception as e:
            logging.exception(e)
            errors_list.append((out_local, [e.output]))
            continue

        if out_local.exists():
            with open(out_local, 'r') as f:
                errors = find_minization_bugs(f.readlines())
        else:
            errors = [out_local + ' doesn\'t exist']

        if errors:
            for e in errors:
                logging.error(e)
            errors_list.append((out_local, errors))
        else:
            minimized_list.append(out_local)
            nmin.remove()

    if errors_list:
        err_file = outdir.joinpath('min_errors.log')
        with open(err_file, 'w') as f:
            for err_pdb, errors in errors_list:
                f.write('\n'.join(['\n', err_pdb] + errors))

    utils.assemble_models(minimized_list, out, remove=True)

    return out, nmin_models, psf


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.DEBUG,
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    cli()
