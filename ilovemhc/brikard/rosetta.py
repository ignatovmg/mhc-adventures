from ilovemhc import define, utils, wrappers
from path import Path
import logging
import click
import pandas as pd
import atom_naming


@click.command()
@click.argument('pdb', type=click.Path(exists=True))
def cli(*args, **kwargs):
    score_models(*args, **kwargs)


def score_models(pdb, out=None, convert=True):
    pdb = Path(pdb)
    if not out:
        out = pdb.dirname().joinpath('rosetta.csv')

    split_dir = pdb.dirname().joinpath('split')
    split_dir.mkdir_p()
    logging.info('Splitting %s into %s' % (pdb, split_dir))

    # convert to Rosetta atom naming
    pdb_converted = Path(wrappers.tmp_file_name('.pdb'))
    if convert:
        atom_naming.convert_to_rosetta(pdb, pdb_converted)
    else:
        Path.copyfile(pdb, pdb_converted)

    # split models into split/ dir
    utils.split_models(pdb_converted, split_dir)
    pdb_converted.remove()

    models = sorted(split_dir.listdir())
    models_list = split_dir.joinpath('models.list')
    models_list.write_lines(models)
    nmodels = len(models)

    # compute rosetta scores
    scores_path = out.dirname().joinpath('score.sc')
    scores_path.remove_p()

    rosetta_models_dir = out.dirname()
    call = [define.FLEXPEPDOCK_EXE,
            '-database', define.ROSETTA_DB,
            '-out:no_nstruct_label',
            '-flexpep_score_only',
            '-in:file::l', models_list,
            '-ignore_zero_occupancy', 'false',
            '-out:file:scorefile', scores_path,
            '-overwrite',
            '-out:path:all', rosetta_models_dir]  # last is where to put models made by rosetta (unrequested)
    try: 
        wrappers.shell_call(call, shell=False)
    except:
        pass

    # clean up
    for f in split_dir.listdir():
        f.remove()
        # remove models which rosetta creates
        rosetta_models_dir.joinpath(f.basename()).remove_p()
    split_dir.removedirs()

    # transform to csv
    scores = pd.read_csv(scores_path, skiprows=1, sep='\s+')
    del scores['SCORE:']

    print nmodels, scores.shape[0]
    assert(nmodels == scores.shape[0])
    scores.to_csv(out, float_format='%.4f')

    return out


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.DEBUG,
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    cli()

