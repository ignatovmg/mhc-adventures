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
from .wrappers import *
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

pdb_slices = {'atomi': slice(6, 11),
              'atomn': slice(12, 16),
              'resn': slice(17, 20),
              'chain': slice(21, 22),
              'resi': slice(22, 26),
              'x': slice(30, 38),
              'y': slice(38, 46),
              'z': slice(46, 54)}

pdb_formats = {'atomi': '%5i',
               'atomn': '%4s',
               'resn': '%3s',
               'chain': '%1s',
               'resi': '%4i',
               'x': '%8.3f',
               'y': '%8.3f',
               'z': '%8.3f'}

# atom_regex = re.compile('^ATOM {2}[0-9 ]{5} .{4}.[A-Z]{3} [A-Z][0-9 ]{4}. {3}[ \-0-9]{4}\.[0-9]{3}[ \-0-9]{4}\.[0-9]{3
# }[ \-0-9]{4}\.[0-9]{3}.{22}.\S')
atom_regex = re.compile('^ATOM {2}[0-9 ]{5} .{4}.[A-Z]{3} [A-Z][0-9 ]{4}. {3}[ \-0-9]{4}\.[0-9]{3}[ \-0-9]{4}\.[0-9]{3}\
[ \-0-9]{4}\.[0-9]{3}')


def check_atom_record(line):
    return atom_regex.match(line)


def change_atom_record(line, **kwargs):
    for key, val in kwargs.items():
        if key == 'atomn':
            val = '%-3s' % val
        line = line[:pdb_slices[key].start] + (pdb_formats[key] % val) + line[pdb_slices[key].stop:]
    return line


def load_gdomains_mhc(pdb_id):
    return prody.parsePDB(define.GDOMAINS_DIR /pdb_id + '_mhc_ah.pdb')


def load_gdomains_peptide(pdb_id):
    return prody.parsePDB(define.GDOMAINS_DIR / pdb_id + '_pep_ah.pdb')


def get_atom_fields(line, *args, **kwargs):
    result = []
    strip_flag = kwargs.get('strip', False)

    for field in args:
        x = line[pdb_slices[field]]
        if field == 'atomi' or field == 'resi':
            x = int(x)
        elif field == 'x' or field == 'y' or field == 'z':
            x = float(x)
        elif strip_flag:
            x = x.strip()
        result.append(x)

    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)


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


def split_models(path, outdir, mdl_format='%06i'):
    path = Path(path)
    outdir = Path(outdir)
    slices = pdb_to_slices(path)

    with open(path, 'r') as f:
        lines = f.readlines()

    model_list = []
    for mdl_count, s in enumerate(slices, 1):
        mdl_lines = lines[s]
        name = outdir.joinpath((mdl_format + '.pdb') % mdl_count)

        if mdl_lines[-1].startswith('ENDMDL'):
            mdl_lines[-1] = 'END\n'

        name.write_text(''.join(mdl_lines))
        model_list.append(name)

    return model_list


def assemble_models(pdb_list, out, remove=False):
    with open(out, 'w') as f:
        for i, pdb in enumerate(pdb_list, 1):
            with open(pdb, 'r') as r:
                lines = filter(lambda x: not (x.startswith('MODEL') or x.startswith('END')), r)
                f.write(''.join(['MODEL %i\n' % i] + lines + ['ENDMDL\n']))
            if remove:
                pdb.remove()
    return out


def pdb_to_slices(pdb):  # TODO: fails if MODEL is the last record
    mdl_lines = []
    end_lines = []

    if isinstance(pdb, str) or isinstance(pdb, Path):
        with open(pdb, 'r') as f:
            lines = f.readlines()
    else:
        lines = list(pdb)

    for linei, line in enumerate(lines):
        if line.startswith('MODEL'):
            mdl_lines.append(linei)

        if line.startswith('END'):
            end_lines.append(linei)

    slices = []
    if not mdl_lines:
        if end_lines:
            if len(end_lines) > 1:
                logger.warning('Multiple models found, but no MODEL records')
                start = 0
                for end in end_lines:
                    slices.append(slice(start, end + 1))
                    start += end + 1
            else:
                slices.append(slice(0, end_lines[0] + 1))
        else:
            logger.warning('Warning no END record found')
            slices.append(slice(0, linei + 1))
    else:
        if len(mdl_lines) != len(end_lines):
            logger.warning('Warning # of MODEL records != # of ENDMDL records')
            if len(end_lines) == 0:
                logger.warning('Warning no ENDMDL records are present at all')

        mi = 0
        ei = 0
        last = None
        while mi < len(mdl_lines) or ei < len(end_lines):
            mdl = mdl_lines[min(mi, len(mdl_lines) - 1)]
            end = end_lines[ei]

            if last is None:
                last = (mdl, 'mdl')
                mi += 1
                continue

            if mdl < end and mi < len(mdl_lines):
                if last[1] == 'mdl':
                    logger.warning('Warning no terminal END is found for MODEL record: line %i. Skipping model' % mdl)

                last = (mdl, 'mdl')
                mi += 1
            else:
                if last[1] == 'mdl':
                    slices.append(slice(last[0], end + 1))
                else:
                    logger.error('Error no MODEL record found between 2 END records: (line %i - line %i). Skipping model' % (last[0], end))
                last = (end, 'end')
                ei += 1

    return slices

            
def _pdb_to_slices_old(pdb):
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
        logger.error('Called process returned ' + str(status))
        # raise RuntimeError('Called process returned ' + str(status))
        
    if remove_user_info:
        output = filter(lambda x: not x.startswith('USER'), output)
        
    output = renumber_pdb(None, output)
        
    if save:
        with open(save, 'w') as f:
            f.write(''.join(output))
        return save
        
    return output

        
def hsd2his(path, out=None):
    if out:
        call = 'sed "s/HSD/HIS/g; s/HSE/HIS/g; s/HSP/HIS/g" %s > %s' % (path, out)
    else:
        call = 'sed -i "s/HSD/HIS/g; s/HSE/HIS/g; s/HSP/HIS/g" %s' % path

    shell_call(call, shell=True)
    return out


def his2hsd(path, out=None):
    parsed = prody.parsePDB(path)
    new_res_dict = {}
    his = parsed.select('resname HIS and name CA')

    if his is not None:
        his_list = zip(his.getResnums(), his.getChids())

        for resi, chain in his_list:
            his_res = parsed.select('resname HIS and resnum {} and chain {}'.format(resi, chain))
            atom_list = his_res.getNames()
            if 'HD1' in atom_list:
                if 'HE2' in atom_list:
                    new_resn = 'HSP'
                else:
                    new_resn = 'HSD'
            else:
                new_resn = 'HSE'

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

    return out


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
    pdb = Path(save_file)

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

    return save_file


# can be pdb path or list of lines
def renumber_pdb(save_file, pdb, keep_resi=True, keep_chain=True):
    if isinstance(pdb, str) or isinstance(pdb, Path):
        with open(pdb, 'r') as f:
            lines = f.readlines()
    else:
        lines = list(pdb)
    new_lines = []

    slices = pdb_to_slices(pdb)
    for s in slices:
        prev_resi = None
        prev_chain = None
        new_resi = 0
        counter = 1

        for line in lines[s]:
            new_line = line
            if line.startswith('ATOM') or line.startswith('HETATM'):
                if not keep_resi:
                    resi, chain = get_atom_fields(line, 'resi', 'chain')

                    if resi != prev_resi:
                        new_resi += 1
                        prev_resi = resi

                    if not keep_chain:
                        new_line = change_atom_record(line, atomi=counter, resi=new_resi, chain='A')
                    else:
                        if chain != prev_chain:
                            new_resi = 1
                            prev_chain = chain
                        new_line = change_atom_record(line, atomi=counter, resi=new_resi)
                else:
                    new_line = change_atom_record(line, atomi=counter)
                counter += 1
            new_lines.append(new_line)

    if save_file:
        save_file = Path(save_file)
        with open(save_file, 'w') as o:
            o.write(''.join(new_lines))
        return save_file
    
    return new_lines


def match_lines(lines, ref_lines):
    old_order = []
    for i, line in enumerate(lines):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom = get_atom_fields(line, 'chain', 'resi', 'resn', 'atomn')
            old_order.append((atom, i))
    old_order = zip(*old_order)
    old_order = pd.Series(old_order[1], index=old_order[0])

    ref_order = []
    for i, line in enumerate(ref_lines):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom = get_atom_fields(line, 'chain', 'resi', 'resn', 'atomn')
            ref_order.append(atom)

    diff = set(list(old_order.index)) ^ set(ref_order)
    if diff:
        raise RuntimeError('Matching atoms in the first model failed:\n%s' % str(diff))

    new_order = old_order[ref_order].values
    old_order = old_order.values
    return old_order, new_order


def match_by_residue_position(pdb, ref, out=None, order_by=None):
    """
    Matches atom names to the reference residuewise
    :param pdb:
    :param ref:
    :return:
    """
    if out is None:
        out = pdb[:-4] + '_matched.pdb'
    out = Path(out)

    pdb_parsed = prody.parsePDB(pdb)
    ref_parsed = prody.parsePDB(ref)
    pdb_hiv = pdb_parsed.getHierView()
    ref_hiv = ref_parsed.getHierView()

    # assume number of residues
    assert(pdb_hiv.numResidues() == ref_hiv.numResidues())

    pdb_rlist = pd.Series(list(pdb_hiv.iterResidues()), index=[r.getResnum() for r in pdb_hiv.iterResidues()])
    ref_rlist = pd.Series(list(ref_hiv.iterResidues()), index=[r.getResnum() for r in ref_hiv.iterResidues()])
    if order_by == 'resi':
        pdb_rlist = pdb_rlist[ref_rlist.index]
    elif isinstance(order_by, list):
        pdb_rlist = pdb_rlist[ref_rlist.index]
    elif order_by is None:
        pass
    else:
        raise ValueError('order_by = %s' % str(order_by))

    new_order = []
    natoms = 0
    for r1, r2 in zip(pdb_rlist, ref_rlist):
        # same residue
        logger.debug('%s - %s' % (r1, r2))
        assert(r1.getResname() == r2.getResname())
        atoms1 = r1.getNames()
        atoms2 = r2.getNames()

        # identical naming
        assert(set(atoms1) == set(atoms2))

        # non-redundant names
        assert(len(set(atoms1)) == len(atoms1))
        s = pd.Series(r1.getIndices(), atoms1)
        new_order += list(s[atoms2].values)

        natoms += len(atoms1)

    coords = pdb_parsed.getCoordsets()
    for set_id in range(coords.shape[0]):
        coords[set_id] = coords[set_id][new_order]

    ref_parsed.setCoords(coords)
    prody.writePDB(out, ref_parsed)

    return out, zip(list(pdb_rlist.index), list(ref_rlist.index))


def match_pdb(pdb, ref, out=None):
    pdb = Path(pdb)
    ref = Path(ref)

    pdb_slice_list = pdb_to_slices(pdb)
    ref_slice_list = pdb_to_slices(ref)
    assert(len(ref_slice_list) == 1)

    # verify the same number of lines in each slice
    assert(len(set([x.stop - x.start for x in pdb_slice_list])) == 1)

    if not out:
        out = pdb
    out = Path(out)

    with open(pdb, 'r') as p, open(ref, 'r') as r:
        models_lines = p.readlines()
        ref_lines = r.readlines()

    first_model_lines = models_lines[pdb_slice_list[0]]
    old_order, new_order = match_lines(first_model_lines, ref_lines)

    with open(out, 'w') as f:
        for model in pdb_slice_list:
            lines = np.array(models_lines[model])
            lines[old_order] = lines[new_order]
            lines = renumber_pdb(None, lines)
            f.writelines(lines)

    return out
            

def prepare_pdb22(pdb,
                  out_prefix,
                  rtf=define.RTF22_FILE,
                  prm=define.PRM22_FILE,
                  change_his=True,
                  remove_tmp=True, 
                  patch_termini=True):

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

        call = [define.PDBNMD_EXE, basename, '--rtf=%s' % rtf, '--prm=%s' % prm, '--psfgen=' + define.PSFGEN_EXE, '--nmin=' + define.NMIN_EXE]
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
    
    return Path(outnmin), Path(outpsf)


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
                coords = get_atom_fields(line, 'x', 'y', 'z')
                loc = dd[key]
                bb_coords[loc:loc+3] = coords
    return bb_coords


def peptide_calc_bb_rsmd(pdb1, pdb2, backbone=True, chain=None, exclude_hydrogens=True):
    names = ['N', 'C', 'CA', 'CB', 'O']
    
    with open(pdb1, 'r') as f:
        lines = filter(lambda x: x.startswith('ATOM') or x.startswith('HETATM'), f.readlines())
        crds1 = {}
        for line in lines:
            if chain and (line[21] != chain):
                continue

            if exclude_hydrogens and get_atom_fields(line, 'atomn')[1] == 'H':
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
    for key, crd1 in crds1.items():
        if backbone:
            if key[1] not in names:
                continue

        #if key[1] == 'CB':
        #    crd2 = crds2.get(key, nanar.copy())
        #else:
        try:
            crd2 = crds2[key]
        except KeyError as e:
            logger.warning('Key Error (%s, %s): ' % (pdb1, pdb2) + str(e))
            continue

        if not np.isnan(crd2).any():
            rmsd += ((crd1 - crd2)**2).sum()
            n += 1
    if n == 0:
        logger.error('No atoms to compute RMSD for (n = 0)')
        return np.nan
    rmsd = np.sqrt(rmsd / n)
    return rmsd


def rmsd_ref_vs_models(ref, models, backbone=False, only_chain_b=True, exclude_hydrogens=True):
    bb_names = ['CA', 'N', 'CB', 'O', 'C']

    ref_slices = pdb_to_slices(ref)
    with open(ref, 'r') as f:
        ref_lines = f.readlines()
        ref_lines = filter(lambda x: x.startswith('ATOM') or x.startswith('HETATM'), ref_lines[ref_slices[0]])
        refcrds = []
        for line in ref_lines:
            if only_chain_b and get_atom_fields(line, 'chain') != 'B':
                continue
                
            aname = get_atom_fields(line, 'atomn')
            if backbone and (aname.strip() not in bb_names):
                continue

            if exclude_hydrogens and aname[1] == 'H':
                continue
                
            coords = np.array(get_atom_fields(line, 'x', 'y', 'z'))
            label = get_atom_fields(line, 'atomn', 'resn', 'resi', strip=True)
            refcrds.append((tuple(label), coords))

    refcrd = dict(refcrds)

    lines = []
    result = []
    mdl_slices = pdb_to_slices(models)
    with open(models, 'r') as f:
        mdl_lines = f.readlines()

    for mi, ms in enumerate(mdl_slices, 1):
        for line in mdl_lines[ms]:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                if only_chain_b and get_atom_fields(line, 'chain') != 'B':
                    continue
                
                aname = get_atom_fields(line, 'atomn')
                if backbone and (aname.strip() not in bb_names):
                    continue

                if exclude_hydrogens and (aname[1] == 'H'):
                    continue
                
                coords = get_atom_fields(line, 'x', 'y', 'z')
                try:
                    coords = np.array(coords)
                except ValueError as e:
                    logger.exception(e)
                    continue
                    
                if any(np.isnan(coords)):
                    logger.warning('Invalid coordinates in %s' % id)
                    continue
                
                label = get_atom_fields(line, 'atomn', 'resn', 'resi', strip=True)
                lines.append((tuple(label), coords))

        n = 0
        rmsd = 0.0
        for label, crd in lines:
            if label in refcrd:
                crd1 = refcrd[label]
                crd2 = crd
                rmsd += ((crd1 - crd2)**2).sum()
                n += 1
            else:
                logger.warning('Model %s Warning: atom %s is not in the reference molecule' % (mi, str(label)))
        if n == 0:
            logger.error('Model %s Error: n = 0' % mi)
            continue
        rmsd = np.sqrt(rmsd/n)
        result.append((mi, rmsd))
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


def compute_sequence_matrix_2lists(seq_list1, seq_list2):
    dmat = np.zeros((len(seq_list1), len(seq_list2)))
    for i in range(len(seq_list1)):
        for j in range(len(seq_list2)):
            val = seq_dist(seq_list1[i], seq_list2[j])
            dmat[i, j] = val
    return dmat


