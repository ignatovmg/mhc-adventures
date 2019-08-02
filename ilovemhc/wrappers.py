import subprocess
import os
import logging


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
    except Exception as e:
        logging.exception(e)
        raise
    
    logging.debug('Command output: ')
    logging.debug(output)
        
    return output


def remove_files(path_list):
    shell_call(['rm', '-f'] + path_list)


def tmp_file_name(ext=''):
    name = ('tmp-%i' % os.getpid()) + ext
    return name
