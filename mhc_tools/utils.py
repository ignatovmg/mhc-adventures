import numpy as np
import pandas as pd
import glob
import re
from itertools import product
from subprocess import Popen, PIPE, STDOUT
from path import Path

import Bio
from Bio.SubsMat import MatrixInfo as matlist
from Bio.pairwise2 import format_alignment
import prody

from . import define
from .define import logger


nielsen_residue_set = [7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97, 99, 114, 116, 118,
                       143, 147, 150, 152, 156, 158, 159, 163, 167, 171]

# HLA-A*01:01:01:01
nielsen_ref_seq = 'GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQKMEPRAPWIEQEGPEYWDQETRNMKAHSQTDRANLGTLRGYYNQSEDGSHTIQIMYG\
CDVGPDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQITKRKWEAVHAAEQRRVYLEGRCVDGLRRYLENGKETLQRTDPPKTHMTHHPISDHEATLRCWALGFYPAEITLTWQR\
DGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWELSSQPTIPIVGIIAGLVLLGAVITGAVVAAVMWRRKSSDRKGGSYTQAASSDSAQGSDVSLTA\
CKV'

contacting_set = [7, 9, 22, 24, 36, 45, 55, 58, 59, 62, 63, 65, 66, 67, 69, 70, 72, 73, 74, 76, 77, 80, 81, 84, 95, 96,
                  97, 99, 114, 116, 117, 118, 123, 124, 133, 143, 144, 146, 147, 148, 150, 151, 152, 153, 155, 156, 157,
                  159, 160, 163, 164, 167, 168, 170, 171, 172]

# pdb: 1ao7
custom_ref_seq = 'GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGC\
DVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQR'


def gdomains_mhc_file(pdb_id):
    return define.GDOMAINS_DIR / pdb_id + '_mhc_ah.pdb'


def load_gdomains_mhc(pdb_id):
    return prody.parsePDB(gdomains_mhc_file(pdb_id))


def gdomains_peptide_file(pdb_id):
    return define.GDOMAINS_DIR / pdb_id + '_pep_ah.pdb'


def load_gdomains_peptide(pdb_id):
    return prody.parsePDB(gdomains_peptide_file(pdb_id))


def global_align(s1, s2):
    aln = Bio.pairwise2.align.globalds(s1, s2, matlist.blosum62, -14.0, -4.0)
    return aln


def __get_square_aln_matrix():
    blosum62 = matlist.blosum62
    alphabet = list(set(list(zip(*blosum62.keys()))[0]))
    matrix_square = {}
    for key, val in blosum62.items():
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
    for a1, a2 in zip(s1, s2):
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
            if resilist[pseudo_counter] == refseq_counter:
                pseudo_counter += 1
                pseq.append(a2)
                if pseudo_counter >= pseqlen:
                    break

    return ''.join(pseq)


def get_pseudo_sequence_nielsen(seq):
    return get_pseudo_sequence(seq, nielsen_ref_seq, nielsen_residue_set)


def get_pseudo_sequence_custom(seq):
    return get_pseudo_sequence(seq, custom_ref_seq, contacting_set)


def convert_allele_name(allele):
    new = allele.split()
    new = 'HLA-' + new[1] if new[0].startswith('HLA') else new[1]
    m = re.match('(.+-[A-Z0-9]+\*[0-9]+:[0-9]+)', new)
    if m:
        return m.group(1)
    return new


def compute_generic_distance_matrix(plist, dfunc):
    dmat = np.zeros((len(plist), len(plist)))
    for i in range(len(plist)):
        for j in range(i, len(plist)):
            val = dfunc(plist[i], plist[j])
            dmat[i, j] = val
            dmat[j, i] = val
    return dmat


def compute_sequence_matrix(seq_list):
    return compute_generic_distance_matrix(seq_list, seq_dist)


def compute_sequence_matrix_2lists(seq_list1, seq_list2):
    dmat = np.zeros((len(seq_list1), len(seq_list2)))
    for i in range(len(seq_list1)):
        for j in range(len(seq_list2)):
            val = seq_dist(seq_list1[i], seq_list2[j])
            dmat[i, j] = val
    return dmat


