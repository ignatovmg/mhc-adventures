import subprocess
import errno
import os
import sys
import numpy as np
import logging

def throw_error(etype, msg):
    raise etype('[ERROR] ' + msg)

def file_is_empty(path):
    file_absent_error(path)
    return os.stat(path).st_size == 0

def file_is_empty_error(path):
    if file_is_empty(path):
        throw_error(OSError, 'File %s is empty' % path)

def file_exists(path):
    return os.path.isfile(path)

def file_absent_error(path):
    if not file_exists(path):
        throw_error(OSError, 'File %s doesn\'t exist' % path)
        
def shell_call(call):
    try:
        logging.debug('Command executed:' + ' '.join(call))
        output = subprocess.check_output(call, stderr=subprocess.STDOUT)
        logging.debug(output)
    except subprocess.CalledProcessError as e:
        logging.exception('Exception caught')
    return output

def remove_files(path_list):
    shell_call(['rm', '-f'] + path_list)
    
def tmp_file_name(ext=''):
    name = ('tmp-%i' % os.getpid()) + ext
    return name