import subprocess
import errno
import os
import sys
import numpy as np

def throw_error(msg):
    sys.stderr.write('[ERROR] ' + msg + '\n')
    sys.exit(1)

def file_is_empty(path):
    file_absent_error(path)
    return os.stat(path).st_size == 0

def file_is_empty_error(path):
    if file_is_empty(path):
        throw_error('File %s is empty' % path)

#def file_exist_error(path):
#    if not file_exists(path):
#        raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), path)

def file_exists(path):
    return os.path.isfile(path)

def file_absent_error(path):
    if not file_exists(path):
        throw_error('File %s doesn\'t exist' % path)
        
def shell_call(call):
    return subprocess.check_output(call, stderr=subprocess.STDOUT)

def remove_files(path_list):
    shell_call(['rm', '-f'] + path_list)
    
def tmp_file_name(ext=''):
    name = ('tmp-%06i' % np.random.randint(0,1000000)) + ext
    return name