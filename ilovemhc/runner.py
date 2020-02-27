import itertools
import logging
from pathlib import Path
import pandas as pd
from collections import namedtuple
import os
import time
import multiprocessing as mp
import shutil
import random
import functools

import generate_peptides
import mhc_peptide

logger = logging.getLogger("default")


def set_df(options):
    df = pd.read_csv(Path(options.csv))
    if options.filters is not None:
        df = df[df[options.filters[0]].isin(options.filters[1:])]
    return df


def get_params(options):
    return [i for i in itertools.product(
        options.rotamers, options.vdw, options.resi
    )]


class Runner(object):
    def __init__(self, options):
        self.wdir = options.wdir
        self.pep_samples_num = options.samples
        self.sampling_params = get_params(options)
        self.rec = options.rec
        self.samples = set_df(options)
        self.template = options.tpl
        
        logger.info("WORKDIR: {}".format(options.wdir))
        logger.info("LOG FILE: {}".format(options.logger_file))


    def generate(self):
        _method_alias = functools.partial(_instance_method_alias, self)
        current_dir = os.getcwd()
        os.chdir(str(self.wdir))
        start_global = time.time()
        pool = mp.Pool(mp.cpu_count() - 1)
        try:
            output_list = pool.map(_method_alias, self.samples.iterrows())
            print output_list
        finally:
            pool.close()
            pool.join()
        end_global = time.time()
        msg_info = "TOTAL: {} ".format(end_global - start_global)
        logger.info(msg_info)
        # with open('processed.txt', 'w') as f:
        #     for item in output_list:
        #         f.write("%s\n" % item)
        os.chdir(current_dir)

    def process(self, sample):
        index, sample = sample[0], sample[1]
        start = time.time()
        pepgen_dir = os.getcwd()
        sample_dir = Path(sample.dir)
        wdir = pepgen_dir / sample_dir
        wdir.mkdir()
        print os.path.abspath(os.curdir)
        os.chdir(str(wdir))
        brikarded_dir = wdir / 'brikard'
        brikarded_dir.mkdir()
        print os.path.abspath(os.curdir)

        rec = sample.mhc_path if self.rec is True else None
        template = self.template if self.template is not None else "9mer.pdb" 
        sampler = generate_peptides.PeptideSampler(pep_seq=sample.peptide, rec=rec, wdir=wdir, template=template)
        for param in self.sampling_params:
            
            start_gen = time.time()
            sampler.generate_peptides(self.pep_samples_num, param[0], param[1], random.randint(1, 1000),
                                      sample_resi_within=param[2], auto_disu=True)
            end_gen = time.time()
            
            spec = "{'rotamers':" + str(param[0]) + ",'vdw':" + str(param[1]) + ",'resi':" + str(param[2]) + "}"
            spec_brikard = spec+"_brikard.pdb"
            
            
            
            spec_rmsd = spec+"_rmsd.txt"
            pep_brikarded_path = wdir / "brikard.pdb"
            pep_brikarded_cache_path = brikarded_dir / spec_brikard
            if not os.path.isfile(str(pep_brikarded_path)):
                logger.error("BRIKARD FAILED WITH ".format(spec+str(sample_dir)))
            else:
                msg_info = "SAMPLE: {} MODELS {} SECONDS: {}".format(sample.dir,spec_brikard, end_gen - start_gen)
                logger.info(msg_info)
                
                shutil.copyfile(str(pep_brikarded_path), str(pep_brikarded_cache_path))
                # TODO: RMSD Calculation
                start_rmsd = time.time()
                pep_orig = mhc_peptide.BasePDB(sample.pep_path)
                pep_brikarded = mhc_peptide.BasePDB(str(pep_brikarded_cache_path))
                rmsd = pep_orig.calc_rmsd_with(pep_brikarded, sel='backbone')
                end_rmsd = time.time()
                
                msg_info = "SAMPLE: {} MODELS {} SECONDS: {}".format(sample.dir, spec_rmsd, end_rmsd - start_rmsd)
                logger.info(msg_info)
                
                with open(str(brikarded_dir/ spec_rmsd), 'w') as f:
                    for r in rmsd[0]:
                        f.write("{}\n".format(r))
                    f.close()

        os.chdir(str(pepgen_dir))
        print os.path.abspath(os.curdir)
        end = time.time()
        msg_info = "SAMPLE: {} SECONDS: {}".format(sample.dir, end - start)
        logger.info(msg_info)
        return str(sample_dir)


def _instance_method_alias(obj, arg):
    return obj.process(arg)


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    Options = namedtuple('Options', [
        'csv',
        'debug',
        'filters',
        'logfile',
        'logger_file',
        'resi',
        'rotamers',
        'samples',
        'silent',
        'vdw',
        'wdir',
        'rec'
    ])
    options = Options(
        'pepgen_3101.csv',
        False,
        ['dir', '2yez_GHAEEYGAETL', '2axf_CPSQEPMSIYVY'],
        'pepgen.log',
        Path('/data/Projects/Laba/LPRJ2/mhc-pepgen/run_test/pepgen.log'),
        [1e-05],
        [1, 2, 3],
        10000,
        False,
        [0.0, 0.1],
        Path('/data/Projects/Laba/LPRJ2/mhc-pepgen/run_test'),
        False
    )
    runner = Runner(options)
    runner.generate()
