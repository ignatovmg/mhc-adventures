import os
import sys
import numpy as np
import pandas as pd
import glob
import shutil
import re
import logging
from itertools import product
import subprocess
from subprocess import Popen, PIPE, STDOUT
from path import Path

import Bio
from Bio.SubsMat import MatrixInfo as matlist
from Bio.pairwise2 import format_alignment
from Bio.SeqUtils import seq3
import prody

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

pdb_slices = {'atomi' : slice(6,11),
              'atomn' : slice(12,16),
              'resn' : slice(17,20),
              'chain' : slice(21,22),
              'resi' : slice(22,26),
              'x' : slice(30,38),
              'y' : slice(38,46),
              'z' : slice(46,54)}

pdb_formats = {'atomi' : '%5i',
               'atomn' : '%4s',
               'resn' : '%3s',
               'chain' : '%1s',
               'resi' : '%4i',
               'x' : '%8.3f',
               'y' : '%8.3f',
               'z' : '%8.3f'}


#atom_regex = re.compile('^ATOM {2}[0-9 ]{5} .{4}.[A-Z]{3} [A-Z][0-9 ]{4}. {3}[ \-0-9]{4}\.[0-9]{3}[ \-0-9]{4}\.[0-9]{3}[ \-0-9]{4}\.[0-9]{3}.{22}.\S')
atom_regex = re.compile('^ATOM {2}[0-9 ]{5} .{4}.[A-Z]{3} [A-Z][0-9 ]{4}. {3}[ \-0-9]{4}\.[0-9]{3}[ \-0-9]{4}\.[0-9]{3}[ \-0-9]{4}\.[0-9]{3}')


def check_atom_record(line):
    return atom_regex.match(line)


def change_atom_record(line, **kwargs):
    for key, val in kwargs.iteritems():
        if key == 'atomn':
            val = '%-3s' % val
        line = line[:pdb_slices[key].start] + (pdb_formats[key] % val) + line[pdb_slices[key].stop:]
    return line


def get_atom_fields(line, *args):
    result = []
    for field in args:
        x = line[pdb_slices[field]]
        if field == 'atomi' or field == 'resi':
            x = int(x)
        result.append(x)
    return tuple(result)


def global_align(s1, s2):
    aln = Bio.pairwise2.align.globalds(s1, s2, matlist.blosum62, -14.0, -4.0)
    return aln


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


def convert_allele_name(allele):
    new = allele.split()
    new = 'HLA-' + new[1] if new[0].startswith('HLA') else new[1]
    m = re.match('(.+-[A-Z0-9]+\*[0-9]+:[0-9]+)', new)
    if m:
        return m.group(1)
    return new


def split_models(path, outdir, add_rec=None, mdl_format='%06i'):
    if add_rec:
        with open(add_rec, 'r') as f:  
            rec_lines = [x for x in f if x.startswith('ATOM') or x.startswith('HETATM')]
            first_atom = int(rec_lines[-1][6:11]) + 1
            rec_text = ''.join(rec_lines)

    mdl_count = 0
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                mdl_count += 1
                # name = os.path.join(outdir, (mdl_format + '.pdb') % int(line.split()[1]))
                name = os.path.join(outdir, (mdl_format + '.pdb') % int(mdl_count))
                f = open(name, 'w')
                if add_rec:
                    counter = 0
                    f.write(rec_text)
                continue
            if line.startswith('END'):
                f.write('END\n')
                f.close()
                continue
            if add_rec and (line.startswith('ATOM') or line.startswith('HETATM')):
                atom_id = first_atom + counter
                line = line[:6] + '%5i' % atom_id + line[11:]
                counter += 1
                 
            f.write(line)

            
def pdb_to_slices(pdb):
    slices = []
    with open(pdb, 'r') as f:
        limit = [None, None]
        
        for i, line in enumerate(f):
            if line.startswith('MODEL'):
                limit[0] = i
                
            if line.startswith('END'):
                limit[1] = i + 1
                
            if limit[1]:
                if not limit[0]:
                    limit[0] = slices[-1].stop if slices else 0
                
                slices.append(slice(*limit))
                limit = [None, None]
             
        if not slices:
            slices.append(slice(0, i+1))

    return slices


def reduce_pdb(pdb, save=None, input_is_path=True, trim_first=True, remove_user_info=True):
    if input_is_path:
        with open(pdb, 'r') as f:
            pdb_lines = f.readlines()
    else:
        pdb_lines = pdb
    
    if trim_first:
        p_start = Popen([define.REDUCE_EXE, '-Quiet', '-Trim', '-'], stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        p_finish = Popen([define.REDUCE_EXE, '-Quiet', '-FLIP', '-'], stdin=p_start.stdout, stdout=PIPE, stderr=STDOUT)
    else:
        p_start = Popen([define.REDUCE_EXE, '-Quiet', '-FLIP', '-'], stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        p_finish = p_start
        
    for line in pdb_lines:
        p_start.stdin.write(line)
        
    p_start.stdin.close()
    
    output = []
    while p_finish.poll() is None:
        output += p_finish.stdout.readlines()
    
    status = p_finish.poll()
    
    if status != 0:
        logging.error('Called process returned ' + str(status))
        # raise RuntimeError('Called process returned ' + str(status))
        
    if remove_user_info:
        output = filter(lambda x: not x.startswith('USER'), output)
        
    output = renumber_pdb(None, output)
        
    if save:
        with open(save, 'w') as f:
            f.write(''.join(output))
        return
        
    return output
    
        
def hsd2his(path, out=None):
    if out:
        call = 'sed "s/HSD/HIS/g; s/HSE/HIS/g; s/HSP/HIS/g" %s > %s' % (path, out)
    else:
        call = 'sed -i "s/HSD/HIS/g; s/HSE/HIS/g; s/HSP/HIS/g" %s' % path
        
    return shell_call(call, shell=True)


def his2hsd(path, out=None):
    parsed = prody.parsePDB(path)
    new_res_dict = {}
    his = parsed.select('resname HIS and name CA')

    if his is not None:
        his_list = zip(his.getResnums(), his.getChids())

        for resi, chain in his_list:
            his_res = parsed.select('resname HIS and resnum {} and chain {}'.format(resi, chain))
            atom_list = his_res.getNames()
            if 'HE2' in atom_list:
                if 'HD1' in atom_list:
                    new_resn = 'HSP'
                else:
                    new_resn = 'HSE'
            else:
                new_resn = 'HSD'
            new_res_dict[(chain, resi)] = new_resn

    if not out:
        out = path

    # Not using prody writer, because I want to keep other records in pdb as well
    with open(path, 'r') as f:
        lines = f.readlines()

    with open(out, 'w') as o:
        for line in lines:
            if line.startswith('ATOM'):
                key = get_atom_fields(line, 'chain', 'resi')
                resn = new_res_dict.get(key, None)
                if resn:
                    o.write(change_atom_record(line, resn=resn))
                    continue
            o.write(line)


def _his2hsd_old(path, out=None):
    # deprecated
    with open(path, 'r') as f:
        lines = f.readlines()
    if not out:
        out = path
    
    resn_s = pdb_slices['resn']
    resi_s = pdb_slices['resi']
    atomn_s = pdb_slices['atomn']
    resi_prev = None
    
    def change_his(lines, start, end):
        atomn_list = [x[atomn_s].strip() for x in lines[start:end]]
        if 'HE2' in atomn_list:
            if 'HD1' in atomn_list:
                new_resn = 'HSP'
            else:
                new_resn = 'HSE'
        else:
            new_resn = 'HSD'
        
        for i in range(start, end):
            lines[i] = change_atom_record(lines[i], resn=new_resn)
        
    start = None
    for i, line in enumerate(lines):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            if line[resn_s] == 'HIS':
                resi = line[resi_s]
                if resi != resi_prev:
                    if start:
                        change_his(lines, start, end)
                    start = i
                    resi_prev = resi
                end = i+1
                
    # change last HIS
    if start is not None:
        change_his(lines, start, end)
            
    with open(out, 'w') as f:
        f.writelines(lines)
        
    return


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


# can be pdb path or list of lines
def renumber_pdb(save_file, pdb, keep_resi=True):
    counter = 1
    
    if isinstance(pdb, str) or isinstance(pdb, Path):
        with open(pdb, 'r') as f:
            lines = f.readlines()
    else:
        lines = list(pdb)
        
    new_lines = []

    prev_resi = None
    prev_chain = None
    new_resi = 0

    for line in lines:
        new_line = line
        if line.startswith('ATOM') or line.startswith('HETATM'):
            resi, chain = int(line[pdb_slices['resi']]), line[pdb_slices['chain']]

            if resi != prev_resi:
                new_resi += 1
                prev_resi = resi

            if chain != prev_chain:
                new_resi = 1
                prev_chain = chain

            if not keep_resi:
                new_line = change_atom_record(line, atomi=counter, resi=new_resi)
            else:
                new_line = change_atom_record(line, atomi=counter)
            counter += 1
        new_lines.append(new_line)

    if save_file:
        with open(save_file, 'w') as o:
            o.write(''.join(new_lines))
        return 
    
    return new_lines


def match_ref_pdb(pdb, ref, out=None, atom_name_mapping=None):
    pdb = Path(pdb)
    ref = Path(ref)

    models_pdb_slices = pdb_to_slices(pdb)
    models_ref_slices = pdb_to_slices(ref)
    assert(len(models_ref_slices) == 1)
    assert(len(set([x.stop - x.start for x in models_pdb_slices])) == 1)

    if not out:
        out = pdb[:-4] + '_matched.pdb'
    out = Path(out)

    with open(pdb, 'r') as p, open(ref, 'r') as r:
        models_lines = p.readlines()
        ref_lines = r.readlines()

    first_model_lines = models_lines[models_pdb_slices[0]]
    chain_s, resi_s, resn_s, atomn_s = pdb_slices['chain'], pdb_slices['resi'], pdb_slices['resn'], pdb_slices['atomn']

    first_model_order = []
    for i, line in enumerate(first_model_lines):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atomn = line[atomn_s]
            if atom_name_mapping:
                atomn = atom_name_mapping[atomn]
            atom = (line[chain_s], line[resi_s], line[resn_s], atomn)
            first_model_order.append((atom, i))
    first_model_order = zip(*first_model_order)
    first_model_order = pd.Series(first_model_order[1], index=first_model_order[0])

    ref_order = []
    for i, line in enumerate(ref_lines):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom = (line[chain_s], line[resi_s], line[resn_s], line[atomn_s])
            ref_order.append(atom)

    diff = set(list(first_model_order.index)) ^ set(ref_order)
    if diff:
        raise RuntimeError('Matching atoms in the rest of the models failed:\n%s' % str(diff))

    old_order = first_model_order.values
    new_order = first_model_order[ref_order].values

    with open(out, 'w') as f:
        for model in models_pdb_slices:
            lines = pd.Series(models_lines[model])
            lines[old_order] = lines[new_order].values
            lines = renumber_pdb(None, lines.sort_index().values)
            f.writelines(lines)
            

def prepare_pdb22(pdb,
                  out_prefix,
                  rtf=define.RTF22_FILE,
                  prm=define.PRM22_FILE,
                  change_his=True,
                  remove_tmp=True, 
                  patch_termini=True):

    #parsed = prody.parsePDB(pdb)
    #parsed.

    pwd = os.getcwd()
    try:

        pdb = Path(pdb).abspath()
        dirname = pdb.dirname()

        rtf = Path(rtf).abspath()
        prm = Path(prm).abspath()

        basename = pdb.basename()
        os.chdir(dirname)

        tmp_file = (basename[:-4] + '-tmp.pdb').lower() #tmp_file_name('.pdb')

        if change_his:
            his2hsd(basename, tmp_file)
        else:
            shutil.copyfile(basename, tmp_file)

        call = [define.PDBPREP_EXE, tmp_file]
        shell_call(call)

        call = [define.PDBNMD_EXE, tmp_file, '--rtf=%s' % rtf, '--prm=%s' % prm]
        #if patch_chains:
        #    call += ['--first', ','.join(['nter'] + [x.lower() for x in patch_chains])]
        #    call += ['--last',  ','.join(['cter'] + [x.lower() for x in patch_chains])]
        if patch_termini:
            call += ['--default-patch']

        call += ['?']
        shell_call(call)

        nmin = tmp_file[:-4] + '_nmin.pdb'
        psf = tmp_file[:-4] + '_nmin.psf'
        file_is_empty_error(nmin)

        outnmin = os.path.join(pwd, out_prefix + '_nmin.pdb')
        outpsf = os.path.join(pwd, out_prefix + '_nmin.psf')
        os.rename(nmin, outnmin)
        os.rename(psf, outpsf)

        if remove_tmp:
            files = tmp_file[:-4] + '-*.????.pdb'
            call = ['rm', '-f'] + glob.glob(files) + [tmp_file]
            shell_call(call)
    except:
        os.chdir(pwd)
        raise
    
    os.chdir(pwd)
    
    return outnmin, outpsf


def get_backbone_coords(pdb, nres):
    names = ['N', 'C', 'CA', 'CB', 'O']

    bb_coords = np.full(nres * len(names) * 3, np.nan)
    dd = dict(zip(product(range(1, nres+1), names), 
                  np.arange(0, bb_coords.shape[0], 3)))
    
    with open(pdb, 'r') as f:
        lines = filter(lambda x: x.startswith('ATOM') or x.startswith('HETATM'), f.readlines())
        crds = {}
        for line in lines:
            key = (int(line[22:26]), line[12:16].strip())
            if key[1] in names:
                coords = [line[30:38], line[38:46], line[46:54]]
                loc = dd[key]
                bb_coords[loc:loc+3] = map(float, coords)
    return bb_coords
    
def peptide_calc_bb_rsmd(pdb1, pdb2, backbone=True, chain=None):
    #print pdb1, pdb2
    names = ['N', 'C', 'CA', 'CB', 'O']
    
    with open(pdb1, 'r') as f:
        lines = filter(lambda x: x.startswith('ATOM') or x.startswith('HETATM'), f.readlines())
        crds1 = {}
        for line in lines:
            if chain and (line[21] != chain):
                continue
                
            coords = [line[30:38], line[38:46], line[46:54]]
            crds1[(line[22:26].strip(), line[12:16].strip())] = np.array(map(float, coords))

    with open(pdb2, 'r') as f:
        lines = filter(lambda x: x.startswith('ATOM') or x.startswith('HETATM'), f.readlines())
        crds2 = {}
        for line in lines:
            if chain and (line[21] != chain):
                continue
                
            coords = [line[30:38], line[38:46], line[46:54]]
            crds2[(line[22:26].strip(), line[12:16].strip())] = np.array(map(float, coords))

    n = 0
    rmsd = 0.0
    nanar = np.array([np.nan]*3)
    for key, crd1 in crds1.iteritems():
        if backbone:
            if key[1] not in names:
                continue

        #if key[1] == 'CB':
        #    crd2 = crds2.get(key, nanar.copy())
        #else:
        try:
            crd2 = crds2[key]
        except KeyError as e:
            logging.warning('Key Error (%s, %s): ' % (pdb1, pdb2) + str(e))
            continue

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
    id = 0
    result = []
    with open(models, 'r') as f:
        for line in f.readlines():
            if line.startswith('MODEL'):
                lines = []
                #id = line.split()[1]
                id += 1
                continue
                
            if line.startswith('ATOM') or line.startswith('HETATM'):
                if only_chain_b and line[21] != 'B':
                    continue
                
                if backbone and (line[12:16].strip() not in bb_names):
                    continue
                
                coords = line[30:38], line[38:46], line[46:54]
                try:
                    coords = np.array(map(float, coords))
                except ValueError as e:
                    logging.exception(e)
                    continue
                    
                if any(np.isnan(coords)):
                    logging.warning('Invalid coordinates in %s' % id)
                    continue
                
                label = map(str.strip, [line[12:16], line[17:20], line[22:26]])
                lines.append((tuple(label), coords))
                
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
                        logging.warning('Model %s Warning: atom %s is not in the reference molecule' % (id, str(label)))
                if n == 0:
                    logging.error('Model %s Error: n = 0' % id)
                    continue
                rmsd = np.sqrt(rmsd/n)
                result.append((id, rmsd))
                lines = []
    return result


def compute_generic_distance_matrix(plist, dfunc):
    dmat = np.zeros((len(plist), len(plist)))
    for i in range(len(plist)):
        for j in range(i, len(plist)):
            val = dfunc(plist[i], plist[j])
            dmat[i, j] = val
            dmat[j, i] = val
    return dmat


def compute_backbone_distance_matrix(plist):
    return compute_generic_distance_matrix(plist, peptide_calc_bb_rsmd)


def compute_sequence_matrix(seq_list):
    return compute_generic_distance_matrix(seq_list, seq_dist)
