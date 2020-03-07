#!/gpfs/projects/KozakovGroup/conda/mignatov/bin/python
#PBS -l nodes=1:ppn=28,walltime=168:00:00
#PBS -N aff_docking
#PBS -q extended
#PBS -j oe
#PBS -o aff_docking.log

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim

from ilovemhc.wrappers import *
from ilovemhc import dataset, template_modelling
from ilovemhc.grids import GridMaker
from ilovemhc.define import GRID_PRM_DIR
from ilovemhc.engines import make_evaluator, get_device
from ilovemhc.brikard import minimization, atom_naming
from ilovemhc.utils import prepare_pdb22

import logging

from path import Path
from multiprocessing import Pool


def _load_cnn(model_file, device_name, ngpu, output_dir, output_prefix):
    for k, v in locals().iteritems():
        logging.info("{:20s} = {}".format(str(k), str(v)))

    logging.info('Getting device..')
    avail, device = get_device(device_name)
    logging.info(device)

    if device_name.startswith('cuda') and not avail:
        logging.warning('CUDA is not available')
        raise RuntimeError('CUDA is not available')

    logging.info('Loading model from %s..' % model_file)
    if str(device) == 'cpu':
        model = torch.load(model_file, map_location='cpu')
    else:
        model = torch.load(model_file)
    logging.info(model)

    # Since DataParallel was used
    model = model.module

    using_data_parallel = False
    if avail and ngpu > 1:
        ngpu_total = torch.cuda.device_count()
        if ngpu_total < ngpu:
            logging.warning('Number of GPUs specified is too large: %i > %i. Using all GPUs' % (ngpu, ngpu_total))
            ngpu = ngpu_total
        if ngpu > 1:
            logging.info('Using DataParallel on %i GPUs' % ngpu)
            model = nn.DataParallel(model, device_ids=list(range(ngpu)))
            using_data_parallel = True

    model.to(device)

    loss = torch.nn.MSELoss()
    logging.info(loss)

    logging.info('Creating evaluator..')
    model_prefix = os.path.basename(model_file)
    evaluator = make_evaluator(model, loss, device, model_dir=output_dir, model_prefix=output_prefix, every_niter=100)

    return evaluator


def _score_models(evaluator, data_csv, data_root, target_column, tag_column, atom_property_csv, bin_size, batch_size, ncores):
    data_table = pd.read_csv(data_csv)
    data_table['target'] = data_table[target_column]
    data_table['tag'] = data_table[tag_column]

    grid_maker = None
    if atom_property_csv:
        logging.info('Getting custom GridMaker..')
        grid_maker = GridMaker(propspath=GRID_PRM_DIR + '/' + atom_property_csv)

    logging.info('Creating data dataset..')
    data_set = dataset.MolDataset(data_table,
                                  data_root,
                                  grid_maker=grid_maker,
                                  bin_size=bin_size,
                                  target_transform=None)

    logging.info('Creating data loader..')
    data_loader = DataLoader(dataset=data_set,
                             batch_size=batch_size,
                             num_workers=ncores,
                             shuffle=False,
                             drop_last=False)

    logging.info('Getting input shape..')
    input_shape = torch.tensor(data_set[0][0].shape).numpy()
    logging.info(input_shape)

    logging.info('Starting evaluator..')
    evaluator(data_loader, data_table, 1, compute_stats=False)

    logging.info("COMPLETED")


def _make_models(outdir, complex_id, modeller, mhcseq, pepseq):
    outdir = Path(outdir)
    peplen = len(pepseq)
    table = modeller.pdb_table
    table = table[np.array([len(x) for x in table.peptide]) == peplen]

    models = []
    for template in table.pdb:
        save = outdir.joinpath('{complex_id}-{template}-{pid}.pdb'.format(complex_id=complex_id, pid=os.getpid(), template=template))
        logging.info('Making %s' % save)
        try:
            modeller.create_model_from_template_scwrl(save, mhcseq, pepseq, template, add_h=True, remove_tmp=True)
            atom_naming.convert_from_rosetta(save)
            models.append(save)
        except Exception as e:
            logging.exception(e)
            logging.error('Failed to create model ' + save)

    return models


def _make_torch_table(outdir, models, tags=None):
    table = pd.DataFrame(list(models), columns=['path'])
    table['tag'] = tags
    table['target'] = 1.0
    path = Path(outdir).joinpath('torch.csv')
    table.to_csv(path)
    return path


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

    modeller = template_modelling.TemplateModeller()
    models = _make_models(outdir, complex_id, modeller, mhcseq, pepseq)

    return zip([complex_id] * len(models), models)


def _get_best_models(args):
    outdir, tag, group, models_list = args
    outdir = Path(outdir)
    logfile = models_list[0][1].dirname().joinpath('{complex_id}-min.log'.format(complex_id=tag))

    _setup_logger(logfile)

    tag = int(tag)
    try:
        best_idx = group.sort_values('prediction', ascending=False).index[0]
        best_path = models_list[best_idx][1]

        with open(best_path, 'r') as f:
            lines = f.readlines()
        best_path.write_lines(lines + ['END\n'])

        best_nmin, best_psf = prepare_pdb22(best_path, best_path.stripext())
        minimized, nmin, psf = minimization.minimize_energy(best_nmin, out=best_nmin[:-8] + 'min.pdb', psf=best_psf,
                                                            clean=True)
        minimized, nmin, psf = Path(minimized), Path(nmin), Path(psf)
        minimized.move(outdir)
        nmin.move(outdir)
        psf.move(outdir)
        return tag, outdir.joinpath(minimized.basename())
    except Exception as e:
        logging.error('Error minimizing model for line %i' % tag)
        logging.exception(e)
        return tag, None


def dock(outdir, complex_ids, mhcseqs, pepseqs, nproc, model_file):
    outdir = Path(outdir)
    outdir.mkdir_p()

    models_dir = Path('models_tmp')
    models_dir.mkdir_p()

    logging.info('Load CNN model')
    cnn_prefix = 'cnn'
    evaluator = _load_cnn(model_file, 'cuda:0', 1, models_dir, cnn_prefix)
    cnn_output = models_dir.joinpath(cnn_prefix + '-output-1.csv')

    args = zip([models_dir] * len(complex_ids), complex_ids, mhcseqs, pepseqs)

    final_table = outdir.joinpath('models.csv')
    final_table.write_text('complex_id,path\n')

    logging.info('Started main loop')
    for epoch, local_args in enumerate([args[x:x+nproc] for x in np.arange(0, len(args), nproc)]):
        logging.info('Epoch %i' % epoch)

        # Cleaning
        for f in models_dir.files():
            f.remove_p()

        models_list = None
        p = Pool(nproc)
        try:
            print(local_args)
            models_list = p.map(_fun, list(local_args))
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

        models_list = reduce(lambda x, y: x + y, models_list)

        torch_table = _make_torch_table(models_dir, zip(*models_list)[1], zip(*models_list)[0])
        _score_models(evaluator, torch_table, '.', 'target', 'tag', None, 1.0, 64, nproc)

        p = Pool(nproc)
        cnn_table = pd.read_csv(cnn_output)
        min_args = [(outdir, tag, group, models_list) for tag, group in cnn_table.groupby('tag')]
        final_models = p.map(_get_best_models, min_args)
        final_table.write_lines(['%i,%s' % (tag, path) for tag, path in final_models], append=True)

    return final_table


def _get_mhc_seq_dict():
    path = Path('/gpfs/projects/KozakovGroup/mhc_learning/analysis/allele_sequences.csv')
    df = pd.read_csv(path, sep=' ')
    return dict(zip(df.allele_short, df.sequence))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.DEBUG,
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    cnn = '/gpfs/projects/KozakovGroup/mhc_learning/cnn/learning/brikarded_set1/models/RegModel1/RegModel1-0.0001-0.001/model-model-11.pth'

    allele2seq = _get_mhc_seq_dict()

    affinity_table = pd.read_csv('/gpfs/projects/KozakovGroup/mhc_learning/analysis/affinity_data/affinity_clean-15012019.csv')
    ninemers = affinity_table[affinity_table.length == 9]
    ninemers.to_csv('local_affinity.csv')

    mhcseqs = [allele2seq[x] for x in ninemers.allele]
    pepseqs = list(ninemers.peptide)
    dock('models', ninemers.index, mhcseqs, pepseqs, 28, cnn)



