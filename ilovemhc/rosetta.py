from ilovemhc import define, wrappers, mhc_peptide
from path import Path
import logging
import pandas as pd
import numpy as np
import json


class FlexPepDock(object):
    def __init__(self, prot):
        self.prot = prot.copy()
        self.scores = None
        self.logs = None

    def make_springs(self,
                     chain='B',
                     nter_crd=(8.770, 10.594, 9.683),
                     cter_crd=(31.514, 10.064, 10.259)):

        sel = self.prot.ag.select('chain ' + chain + ' and name CA')
        resis = sel.getResnums()
        nter = ('CA', resis[0], chain)
        cter = ('CA', resis[-1], chain)
        fmt = 'CoordinateConstraint {} {}{} CA 1A {:.3f} {:.3f} {:.3f} HARMONIC 0.0 0.5\n'

        springs = fmt.format(nter[0], nter[1], nter[2], *nter_crd) + fmt.format(cter[0], cter[1], cter[2], *cter_crd)
        return springs

    def fire(self, constraints=False, wdir='.', clean=True, flexpepdock_prms=[]):
        """
        Score models using rosetta flexpepdock.
        """

        wdir = Path(wdir)
        input_dir = wdir.joinpath('rosetta_input')
        output_dir = wdir.joinpath('rosetta_output')
        wrappers.rmdir(input_dir)
        wrappers.rmdir(output_dir)
        input_dir.mkdir_p()
        output_dir.mkdir_p()

        # convert to Rosetta atom naming
        logging.info('Converting name to Rosetta')
        self.prot.to_rosetta()

        # save models separately
        flist = self.prot.save_sep(input_dir)

        # compute rosetta scores
        scores_path = output_dir.joinpath('score.sc')
        scores_path.remove_p()

        call = [define.FLEXPEPDOCK_EXE,
                '-database', define.ROSETTA_DB,
                '-out:no_nstruct_label',
                #'-in:file::l', models_list,
                '-ignore_zero_occupancy', 'false',
                #'-out:file:scorefile', scores_path.basename(),
                '-overwrite',
                '-out:path:all', output_dir]

        call.append('-flexPepDockingMinimizeOnly')

        springs_file = output_dir.joinpath('springs')
        if constraints:
            springs_file.write_text(self.make_springs())
            call += ['-constraints::cst_fa_file', springs_file]
            call += ['-constraints:cst_fa_weight', '10.0']

        #call.append('-flexpep_score_only')

        call += flexpepdock_prms

        self.logs = {}
        for pdb_in in flist:
            try:
                output = wrappers.shell_call(call + ['-in:file:s', pdb_in], shell=False)
            except Exception as e:
                output = 'Error minimizing ' + pdb_in
                logging.error(output)
            self.logs[str(Path(pdb_in).basename())] = output

        # transform to csv
        scores = pd.read_csv(scores_path, skiprows=1, sep='\s+')
        del scores['SCORE:']

        # reorder by model id
        scores = scores.iloc[np.argsort(map(int, scores['description'])), :]
        scores_csv = output_dir.joinpath('scores.csv')
        scores.to_csv(scores_csv, float_format='%.4f')

        # read generated models
        min_list = [output_dir.joinpath(str(model) + '.pdb') for model in scores['description']]
        agmin = mhc_peptide.BasePDB(pdb_list=min_list)
        agmin.save('min_tmp.pdb')
        result = mhc_peptide.BasePDB('min_tmp.pdb')

        if clean:
            wrappers.rmdir(input_dir)
            wrappers.rmdir(output_dir)
            wrappers.remove_files(['min_tmp.pdb', scores_csv])
            springs_file.remove_p()

        self.scores = scores
        self.prot = result
        self.prot.from_rosetta()

    def save(self, dirname='.', basename='rosmin'):
        dirname = Path(dirname)
        dirname.mkdir_p()
        self.prot.save(dirname.joinpath(basename + '.pdb'))
        self.scores.to_csv(dirname.joinpath(basename + '.csv'), float_format='%.4f')
        with open(dirname.joinpath(basename + '_log.json'), 'w') as f:
            json.dump(self.logs, f, indent=4)


