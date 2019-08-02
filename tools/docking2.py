#!/gpfs/projects/KozakovGroup/conda/mignatov/bin/python
#PBS -l nodes=1:ppn=28,walltime=168:00:00
#PBS -N aff_docking
#PBS -q extended
#PBS -j oe
#PBS -o aff_docking.log

import pandas as pd
import sys

from ilovemhc import template_modelling
from ilovemhc.brikard import minimization, atom_naming
from ilovemhc.utils import prepare_pdb22

import logging
from path import Path
from multiprocessing import Pool


allele_sequences = '/gpfs/projects/KozakovGroup/mhc_learning/analysis/allele_sequences.csv'


def _setup_logger(log_file):
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logger.setLevel(logging.DEBUG)
    fileh = logging.FileHandler(log_file, 'w')
    fileh.setFormatter(formatter)
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    logger.addHandler(fileh)


def _fun(args):
    outdir, complex_id, mhcseq, pepseq = args
    outdir = Path(outdir)
    logfile = outdir.joinpath('{complex_id}.log'.format(complex_id=complex_id))

    _setup_logger(logfile)

    try:
        modeller = template_modelling.TemplateModeller()
        template = modeller.pick_template(mhcseq, pepseq)
        save = outdir.joinpath('{complex_id}-{template}.pdb'.format(complex_id=complex_id, template=template))
        modeller.create_model_from_template_scwrl(save, mhcseq, pepseq, template)
        atom_naming.convert_from_rosetta(save)

        nmin, psf = prepare_pdb22(save, save.stripext())
        minimized, nmin, psf = minimization.minimize_energy(nmin, out=nmin[:-8] + 'min.pdb', psf=psf, clean=True)

        output = (complex_id, Path(minimized).basename(), Path(nmin).basename(), Path(psf).basename())
        logging.info('TABLE_INDEX %i,%s,%s,%s' % output)

        #nmin.remove_p()
        save.remove_p()
    except Exception as e:
        logging.exception(e)
        logging.error('Failed to create model ' + str(complex_id))

        output = (complex_id, None, None, None)
        logging.info('TABLE_INDEX %i,%s,%s,%s' % output)

    return output


def dock(outdir, complex_ids, mhcseqs, pepseqs, nproc):
    outdir = Path(outdir)
    outdir.mkdir_p()

    args = zip([outdir] * len(complex_ids), complex_ids, mhcseqs, pepseqs)

    #final_table = outdir.joinpath('models.csv')
    #final_table.write_text('complex_id,min,nmin,psf\n')

    logging.info('Started main loop')
    models_list = None
    p = Pool(nproc)
    try:
        models_list = p.map(_fun, list(args))
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        print("Normal termination")
        p.close()
    p.join()

    if models_list is None:
        logging.info('Template generation failed')
        return None

    #print(models_list)
    #final_table.write_lines(['%i,%s,%s,%s' % tuple(x) for x in models_list], append=True)
    #return final_table


def _get_mhc_seq_dict():
    path = Path(allele_sequences)
    df = pd.read_csv(path, sep=' ')
    return dict(zip(df.allele_short, df.sequence))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.DEBUG,
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    try:
        output_dir = sys.argv[1]
        table = pd.read_csv(sys.argv[2])
        nproc = int(sys.argv[3])
    except:
        print('\nUsage: %s outdir table.csv number_of_processors\n' % __file__)
        raise

    allele2seq = _get_mhc_seq_dict()
    mhcseqs = [allele2seq[x] for x in table.allele]
    pepseqs = list(table.peptide)
    line_ids = list(table.index)

    dock(output_dir, line_ids, mhcseqs, pepseqs, 28)
