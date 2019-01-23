import os
import sys
import numpy as np
import pandas as pd
import glob
import shutil

import Bio
from Bio.SubsMat import MatrixInfo as matlist
from Bio.pairwise2 import format_alignment
from Bio.SeqUtils import seq3

import define
from wrappers import *

nielsen_residue_set = [7, 9, 24, 45, 59, 62, 63, 66, 67, 
            69, 70, 73, 74, 76, 77, 80, 81, 
            84, 95, 97, 99, 114, 116, 118, 143, 
            147, 150, 152, 156, 158, 159, 163, 167, 171]

# HLA-A*01:01:01:01
nielsen_ref_seq = 'GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQKMEPRAPWIEQEGPEYWDQETRNMKAHSQTDRANLGTLRGYYNQSEDGSHTIQIMYGCDVGPDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQITKRKWEAVHAAEQRRVYLEGRCVDGLRRYLENGKETLQRTDPPKTHMTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWELSSQPTIPIVGIIAGLVLLGAVITGAVVAAVMWRRKSSDRKGGSYTQAASSDSAQGSDVSLTACKV'

contacting_set = [7, 9, 22, 24, 36, 45, 55, 58, 59, 62, 63, 65, 66, 67, 69, 
              70, 72, 73, 74, 76, 77, 80, 81, 84, 95, 96, 97, 99, 114, 
              116, 117, 118, 123, 124, 133, 143, 144, 146, 147, 148, 150, 
              151, 152, 153, 155, 156, 157, 159, 160, 163, 164, 167, 168, 170, 171, 172]

# pdb: 1ao7
custom_ref_seq = 'GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQR'

def __get_square_aln_matrix():
    blosum62 = matlist.blosum62
    alphabet = list(set(zip(*blosum62.keys())[0]))
    matrix_square = {}
    for key, val in blosum62.iteritems():
        matrix_square[key] = val
        matrix_square[(key[1],key[0])] = val

    worst_value = min(blosum62.values()) # =-4
    worst_value -= 2
    for a in alphabet:
        matrix_square[(a, '-')] = worst_value
        matrix_square[('-', a)] = worst_value
    matrix_square[('-', '-')] = 0
    return matrix_square

__aln_matrix_square = __get_square_aln_matrix()

def ungapped_score(s1, s2):
    if len(s1) != len(s2):
        return np.nan
    score = 0.0
    for a1,a2 in zip(s1, s2):
        score += __aln_matrix_square[(a1, a2)]
    return score

def seq_dist(s1, s2):
    score = ungapped_score(s1, s2)
    n1 = ungapped_score(s1, s1)
    n2 = ungapped_score(s2, s2)
    d = 1.0 - score / np.sqrt(n1*n2)
    return d

def get_pseudo_sequence(seq, refseq, resilist):
    aln_matrix = matlist.blosum62
    aln = Bio.pairwise2.align.globalds(refseq, seq, aln_matrix, -14.0, -4.0)[0]
    aln1 = aln[0]
    aln2 = aln[1]

    pseudo_counter = 0
    refseq_counter = 0
    pseqlen = len(resilist)
    pseq = [''] * pseqlen
    for a1, a2 in zip(aln1, aln2):
        if a1 != '-':
            refseq_counter += 1
            if contacting_set[pseudo_counter] == refseq_counter:
                pseudo_counter += 1
                pseq.append(a2)
                if pseudo_counter >= pseqlen:
                    break

    return ''.join(pseq)

def get_pseudo_sequence_nielsen(seq):
    return get_pseudo_sequence(seq, nielsen_ref_seq, nielsen_residue_set)

def get_pseudo_sequence_custom(seq):
    return get_pseudo_sequence(seq, custom_ref_seq, contacting_set)

def split_models(path):
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                    name = line.split()[1] + '.pdb'
                    f = open(name, 'w')
                    continue
            if line.startswith('END'):
                    f.write('END\n')
                    f.close()
                    continue
            f.write(line)
            
def hsd2his(path, out=None):
    if out:
        call = 'sed "s/HSD/HIS/g" %s > %s' % (path, out)
    else:
        call = 'sed -i "s/HSD/HIS/g" %s' % path
        
    return os.system(call)

def his2hsd(path, out=None):
    if out:
        call = 'sed "s/HIS/HSD/g" %s > %s' % (path, out)
    else:
        call = 'sed -i "s/HIS/HSD/g" %s' % path
        
    return os.system(call)
            
def merge_two(save_file, pdb1, pdb2, keep_chain=False, keep_residue_numbers=False):
    pdb = save_file

    with open(pdb1, 'r') as f:
        lines1 = f.readlines()

    with open(pdb2, 'r') as f:
        lines2 = f.readlines()

    atomi = 1
    resi = 0
    f = open(pdb, 'w')
    prevresi = ''
    for linei, line in enumerate(lines1 + lines2):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            if line[22:26] != prevresi:
                    prevresi = line[22:26]
                    resi += 1

            newline = line[:6] + "%5i " % atomi + line[12:21]
            if keep_chain:
                newline += line[21]
            else:
                newline += 'A' 
            if keep_residue_numbers:
                newline += line[22:26]
            else:
                newline += "%4i" % resi
            newline += line[26:]

            f.write(newline)
            atomi += 1

        if linei == len(lines1) - 1:
            prevresi = ''
            
    f.write('END\n')
    f.close()

def prepare_pdb22(pdb, out_prefix, rtf=define.RTF22_FILE, prm=define.PRM22_FILE, change_his=True, remove_tmp=True):
    tmp_file = pdb[:-4] + '-tmp.pdb' #tmp_file_name('.pdb')
    
    if change_his:
        his2hsd(pdb, tmp_file)
    else:
        shutil.copyfile(pdb, tmp_file)
    
    call = [define.PDBPREP_EXE, tmp_file]
    shell_call(call)
    
    call = [define.PDBNMD_EXE, tmp_file, '--rtf=%s' % rtf, '--prm=%s' % prm, '?']
    shell_call(call)
    
    nmin = tmp_file[:-4] + '_nmin.pdb'
    psf = tmp_file[:-4] + '_nmin.psf'
    file_is_empty_error(nmin)
    
    outnmin = out_prefix + '_nmin.pdb'
    outpsf = out_prefix + '_nmin.psf'
    os.rename(nmin, outnmin)
    os.rename(psf, outpsf)
    
    if remove_tmp:
        files = tmp_file[:-4] + '-*.????.pdb'
        call = ['rm', '-f'] + glob.glob(files) + [tmp_file]
        shell_call(call)
        
    return outnmin, outpsf
    
def peptide_calc_bb_rsmd(pdb1, pdb2):
    #print pdb1, pdb2
    names = ['N', 'C', 'CA', 'CB', 'O']

    with open(pdb1, 'r') as f:
            lines = filter(lambda x: x.startswith('ATOM') or x.startswith('HETATM'), f.readlines())
            crds1 = {}
            for line in lines:
                    coords = [line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]
                    crds1[(line[22:26], line[12:16])] = np.array(map(float, coords))

    with open(pdb2, 'r') as f:
            lines = filter(lambda x: x.startswith('ATOM') or x.startswith('HETATM'), f.readlines())
            crds2 = {}
            for line in lines:
                    coords = [line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]
                    crds2[(line[22:26], line[12:16])] = np.array(map(float, coords))

    n = 0
    rmsd = 0.0
    nanar = np.array([np.nan]*3)
    for key, crd1 in crds1.iteritems():
        if key[1] not in names:
            continue

        if key[1] == 'CB':
            crd2 = crds2.get(key, nanar.copy())
        else:
            crd2 = crds2[key]

        if not np.isnan(crd2).any():
            rmsd += ((crd1 - crd2)**2).sum()
            n += 1
    rmsd = np.sqrt(rmsd / n)
    return rmsd

def rmsd_ref_vs_models(ref, models, backbone=False, only_chain_b=True):
    bb_names = ['CA', 'N', 'CB', 'O', 'C']
    
    with open(ref, 'r') as f:
        reflines = filter(lambda x: x.startswith('ATOM') or x.startswith('HETATM'), f.readlines())
        refcrds = []
        for line in reflines:
            if only_chain_b and line[21] != 'B':
                continue
                
            if backbone and (line[12:16].strip() not in bb_names):
                continue
                
            coords = [line[30:38], line[38:46], line[46:54]]
            label  = map(str.strip, [line[12:16], line[17:20], line[22:26]])
            refcrds.append((tuple(label), np.array(map(float, coords))))

    refcrd = dict(refcrds)

    lines = []
    id = 1
    with open(models, 'r') as f:
        for line in f.readlines():
            if line.startswith('MODEL'):
                lines = []
                id = line.split()[1]
                continue
            if line.startswith('ATOM') or line.startswith('HETATM'):
                if only_chain_b and line[21] != 'B':
                    continue
                
                if backbone and (line[12:16].strip() not in bb_names):
                    continue
                
                coords = line[30:38], line[38:46], line[46:54]
                label  = map(str.strip, [line[12:16], line[17:20], line[22:26]])
                lines.append((tuple(label), np.array(map(float, coords))))
            if line.startswith('END'):
                rmsd = 0.0
                n = 0
                for label, crd in lines:


                    if label in refcrd:
                        crd1 = refcrd[label]
                        crd2 = crd
                        #print crd1, crd2
                        rmsd += ((crd1 - crd2)**2).sum()
                        n += 1
                    else:   
                        sys.stderr.write('Model %s Warning: atom %s is not in the reference molecule\n' % (id, str(label)))
                if n == 0:
                    sys.stderr.write('Model %s Error: n = 0\n' % id)
                    continue #sys.exit(1)
                rmsd = np.sqrt(rmsd/n)
                print('%s %.3f' % (id, rmsd))
                lines = []
                
def conv3d_output_shape(h_w_d, kernel_size=1, stride=1, pad=0, dilation=1, ceil_flag=False):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    from math import floor, ceil
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size, kernel_size)
        
    rnd = ceil if ceil_flag else floor
    h = rnd(((h_w_d[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = rnd(((h_w_d[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    d = rnd(((h_w_d[2] + (2 * pad) - ( dilation * (kernel_size[2] - 1) ) - 1 )/ stride) + 1)
    return h, w, d