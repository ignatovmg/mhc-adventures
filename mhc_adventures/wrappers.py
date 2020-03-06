import subprocess
import os
import logging
import contextlib
import tempfile
from path import Path


@contextlib.contextmanager
def isolated_filesystem(dir=None, remove=True):
    """A context manager that creates a temporary folder and changes
    the current working directory to it for isolated filesystem tests.
    """
    cwd = os.getcwd()
    if dir is None:
        t = tempfile.mkdtemp(prefix='pocketdock-')
    else:
        t = dir
    os.chdir(t)
    try:
        yield t
    except Exception as e:
        logging.error('Error occured, temporary files are in ' + t)
        raise
    else:
        os.chdir(cwd)
        if remove:
            Path(t).rmtree_p()
    finally:
        os.chdir(cwd)


def file_is_empty(path):
    file_absent_error(path)
    return os.stat(path).st_size == 0


def file_is_empty_error(path):
    if file_is_empty(path):
        raise OSError('File %s is empty' % path)


def file_exists(path):
    return os.path.isfile(path)


def file_absent_error(path):
    if not file_exists(path):
        raise OSError('File %s doesn\'t exist' % path)


def valid_file(path):
    file_is_empty_error(path)


def shell_call(call, shell=False, *args, **kwargs):
    cmd_string = call
    if not shell:
        cmd_string = ' '.join(cmd_string)

    try:
        logging.debug('Command executed: ' + cmd_string)
        output = subprocess.check_output(call, shell=shell, *args, **kwargs)
    except subprocess.CalledProcessError as e:
        logging.exception(e)
        logging.debug(e.output)
        raise
    
    logging.debug('Command output: ')
    logging.debug(output)
        
    return output


def remove_files(path_list):
    shell_call(['rm', '-f'] + path_list)


def tmp_file_name(ext=''):
    name = ('tmp-%i' % os.getpid()) + ext
    return name


def rmdir(dirname):
    Path(dirname).rmtree_p()
