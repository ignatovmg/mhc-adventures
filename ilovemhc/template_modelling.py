import utils
import define
from wrappers import *

import pandas as pd
import numpy as np

import Bio
from Bio.SubsMat import MatrixInfo as matlist
from Bio.pairwise2 import format_alignment
from Bio.SeqUtils import seq3

import pymol
from pymol import cmd

import subprocess as sbs
import logging

class TemplateModeller():
    def __init__(self, pdb_table=None, pdb_path=define.GDOMAINS_DIR, scwrl_bin=define.SCWRL_EXE):
        '''
        pdb_table - pandas table containing columns: 'pdb', 'peptide', 'resi_orig', 'seq_orig', 'pseudo_nielsen'
        '''
        if pdb_table:
            self.pdb_table = pdb_table.copy()
        else:
            self.pdb_table = pd.read_csv(define.TEMPLATE_MODELLER_DEFAULT_TABLE)
            
        self.pdb_path = pdb_path
        
        file_absent_error(scwrl_bin)
        self.scwrl_bin = scwrl_bin

    def _get_mhc_path(self, pdb):
        return self.pdb_path + '/' + pdb + '_mhc_fix_ph.pdb'
    
    def _get_pep_path(self, pdb):
        return self.pdb_path + '/' + pdb + '_pep_fix_ph.pdb'
        
    # the choice of coefficient in scoring for template picking was
    # studied and 0.2 came out the best
    def pick_template(self, mhcseq, pepseq, coef=0.2):
        table = self.pdb_table[self.pdb_table.pep_len == len(pepseq)]
        
        pseudoseq = utils.get_pseudo_sequence_nielsen(mhcseq)
        mhc_dist = np.array([utils.seq_dist(pseudoseq, x) for x in table.pseudo_nielsen])
        pep_dist = np.array([utils.seq_dist(pepseq, x) for x in table.peptide])
        pdbid = table.pdb.iloc[np.argmin(mhc_dist * (1. - coef) + pep_dist * coef)]
        
        return pdbid
        
    def create_model_from_template_pymol(self, savepath, mhcseq, pepseq, pdb, add_polar_h=False):
        table = self.pdb_table
        row = table[table.pdb == pdb].iloc[0, :]
        resi_list = row['resi_orig'].split(',')
        tplseq = row['seq_orig']
        pepseq_orig = row['peptide']

        aln = Bio.pairwise2.align.globalds(mhcseq, tplseq, matlist.blosum62, -14.0, -4.0)[0]
        aln1 = aln[0]
        aln2 = aln[1]
        
        logging.info('Alignment:')
        logging.info('new sequence: ' + aln1)
        logging.info('old sequence: ' + aln2)
        logging.info('')
        logging.info('new peptide: ' + pepseq)
        logging.info('old peptide: ' + pepseq_orig)
        
        mhc_mutations = []
        for anew, aold, resi in zip(list(aln1), list(aln2), resi_list):
            if anew != aold:
                if anew != '-' and anew != 'X' and aold != '-' and aold != 'X':
                    mhc_mutations.append((seq3(anew), seq3(aold), resi))
        logging.info('Found %i mutations in MHC' % len(mhc_mutations))
        
        pep_mutations = []
        pep_resi = map(str, range(len(pepseq_orig)))
        for anew, aold, resi in zip(list(pepseq), list(pepseq_orig), pep_resi):
            if anew != aold:
                if anew != '-' and anew != 'X' and aold != '-' and aold != 'X':
                    pep_mutations.append((anew, aold, resi))

        logging.info('Found %i mutations in peptide' % len(pep_mutations))
        
        cmd.reinitialize()
        cmd.load(self._get_mhc_path(pdb), 'mhc')
        cmd.load(self._get_pep_path(pdb), 'pep')
        
        cmd.wizard('mutagenesis')
        cmd.refresh_wizard()
        cmd.remove("not alt ''+A")
        cmd.alter('all', "alt=''")
        
        for new, old, resi in mhc_mutations:
            logging.info(resi, old, new)
            cmd.get_wizard().do_select("A/" + resi + "/")
            cmd.get_wizard().set_mode(seq3(new).upper())
            cmd.get_wizard().apply()
            
        for new, old, resi in pep_mutations:
            logging.info(resi, old, new)
            cmd.get_wizard().do_select("B/" + resi + "/")
            cmd.get_wizard().set_mode(seq3(new).upper())
            cmd.get_wizard().apply()

        cmd.set_wizard()
        
        if add_polar_h:
            cmd.h_add('donors or acceptors')
                   
        cmd.save(savepath, "all")
        cmd.delete("all")
        
    def create_model_from_template_scwrl(self, savepath, mhcseq, pepseq, pdb, add_h=True, remove_tmp=True):
        table = self.pdb_table
        row = table[table.pdb == pdb].iloc[0, :]
        mhcseq_orig = row['seq_orig']
        pepseq_orig = row['peptide']

        aln = Bio.pairwise2.align.globalds(mhcseq, mhcseq_orig, matlist.blosum62, -14.0, -4.0)[0]
        aln1 = aln[0]
        aln2 = aln[1]
        
        logging.info('Alignment:')
        logging.info('new sequence: ' + aln1)
        logging.info('old sequence: ' + aln2)
        logging.info('')
        logging.info('new peptide: ' + pepseq)
        logging.info('old peptide: ' + pepseq_orig)
        
        mhc_mutations = []
        n_mhc = 0
        for anew, aold in zip(list(aln1), list(aln2)):
            if aold != '-':
                if anew == '-' or anew == aold or anew == 'X':
                    mhc_mutations.append(aold.lower())
                else:
                    mhc_mutations.append(anew)
                    n_mhc += 1
        logging.info('Found %i mutations in MHC' % n_mhc)
                    
        pep_mutations = []
        n_pep = 0
        for anew, aold in zip(list(pepseq), list(pepseq_orig)):
            if aold == anew:
                pep_mutations.append(aold.lower())
            else:
                pep_mutations.append(anew)
                n_pep += 1
        logging.info('Found %i mutations in peptide' % n_pep)

        # merge peptide and mhc
        tmp_pdb_name = str(os.getpid()) + '_merged.pdb'
        utils.merge_two(tmp_pdb_name, self._get_mhc_path(pdb), self._get_pep_path(pdb))
        
        # write down the new sequence
        scwrl_seq = ''.join(mhc_mutations + pep_mutations)
        tmp_seq_name = str(os.getpid()) + '_seq.txt'
        with open(tmp_seq_name, 'w') as f:
            f.write(scwrl_seq + '\n')
            
        # run scwrl
        tmp_out_name = str(os.getpid()) + '_out.pdb'
        call = [self.scwrl_bin, '-s', tmp_seq_name, '-i', tmp_pdb_name, '-o', tmp_out_name]
        if not add_h:
              call += ['-h']
        shell_call(call)
        
        # change peptide chain to B and renumber
        counter = 1
        old_pep_resi = range(len(mhcseq_orig)+1, len(mhcseq_orig)+len(pepseq_orig)+1)
        new_pep_resi = range(1,len(pepseq_orig)+1)
        dic_pep_resi = dict(zip(old_pep_resi, new_pep_resi))
        with open(tmp_out_name, 'r') as f, open(savepath, 'w') as o:
            for line in f:
                new_line = line
                if line.startswith('ATOM'):
                    resi = int(line[22:26])
                    if resi in dic_pep_resi.keys():
                        new_resi = '%4i' % dic_pep_resi[resi]
                        new_line = line[:21] + 'B' + new_resi + new_line[26:]
                o.write(new_line)
                
        if remove_tmp:
            remove_files([tmp_seq_name, tmp_pdb_name, tmp_out_name])
            
        return savepath
    
    def create_model(self, savepath, mhcseq, pepseq, add_h=True, charmm22=True, scwrl=True):
        if savepath[-4:] != '.pdb':
            raise RuntimeError('Output file must have ".pdb" extension')
        
        pdb = self.pick_template(mhcseq, pepseq)
        logging.info('TEMPLATE: %s' % pdb)
        
        if scwrl:
            self.create_model_from_template_scwrl(savepath, mhcseq, pepseq, pdb, add_h, remove_tmp=True)
        else:
            self.create_model_from_template_pymol(savepath, mhcseq, pepseq, pdb, add_h)
        
        outfiles = [pdb, savepath]
        if charmm22:
            outfiles += utils.prepare_pdb22(savepath, savepath[:-4])
        
        return outfiles
