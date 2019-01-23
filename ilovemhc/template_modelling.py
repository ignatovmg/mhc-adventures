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

class TemplateModeller():
    def __init__(self, pdb_table, pdb_path=define.GDOMAINS_DIR):
        self.pdb_table = pdb_table.copy()
        self.pdb_path = pdb_path

    def pick_template(self, mhcseq, pepseq):
        table = self.pdb_table
        pseudoseq = utils.get_pseudo_sequence_nielsen(mhcseq)
        mhc_dist = np.array([utils.seq_dist(pseudoseq, x) for x in table.pseudo_nielsen])
        pep_dist = np.array([utils.seq_dist(pepseq, x) for x in table.peptide])
        sum_dist = sorted(zip(mhc_dist + pep_dist, table.pdb))
        
        best = sum_dist[0]
        pdbid = best[1]
        
        return pdbid
        
    def create_model_from_template(self, savepath, mhcseq, pepseq, pdb, add_polar_h=False):
        table = self.pdb_table
        row = table[table.pdb == pdb].iloc[0, :]
        resi_list = row['resi_orig'].split(',')
        tplseq = row['seq_orig']
        pepseq_orig = row['peptide']

        aln = Bio.pairwise2.align.globalds(mhcseq, tplseq, matlist.blosum62, -14.0, -4.0)[0]
        aln1 = aln[0]
        aln2 = aln[1]
        
        print(aln1)
        print(aln2)

        mhc_mutations = []
        for anew, aold, resi in zip(list(aln1), list(aln2), resi_list):
            if anew != aold:
                if anew != '-' and anew != 'X' and aold != '-' and aold != 'X':
                    mhc_mutations.append((seq3(anew), seq3(aold), resi))
                    
        print('Found %i mutations in MHC' % len(mhc_mutations))
        
        pep_mutations = []
        pep_resi = map(str, range(len(pepseq_orig)))
        for anew, aold, resi in zip(list(pepseq), list(pepseq_orig), pep_resi):
            if anew != aold:
                if anew != '-' and anew != 'X' and aold != '-' and aold != 'X':
                    pep_mutations.append((anew, aold, resi))

        print('Found %i mutations in peptide' % len(pep_mutations))
        
        cmd.reinitialize()
        cmd.load(self.pdb_path + '/' + pdb + '_mhc.pdb', 'mhc')
        cmd.load(self.pdb_path + '/' + pdb + '_pep.pdb', 'pep')
        
        cmd.wizard('mutagenesis')
        cmd.refresh_wizard()
        cmd.remove("not alt ''+A")
        cmd.alter('all', "alt=''")
        
        for new, old, resi in mhc_mutations:
            print(resi, old, new)
            cmd.get_wizard().do_select("A/" + resi + "/")
            cmd.get_wizard().set_mode(seq3(new).upper())
            cmd.get_wizard().apply()
            
        for new, old, resi in pep_mutations:
            print(resi, old, new)
            cmd.get_wizard().do_select("B/" + resi + "/")
            cmd.get_wizard().set_mode(seq3(new).upper())
            cmd.get_wizard().apply()

        cmd.set_wizard()
        
        if add_polar_h:
            cmd.h_add('donors or acceptors')
                   
        cmd.save(savepath, "all")
        cmd.delete("all")
    
    def create_model(self, savepath, mhcseq, pepseq, add_polar_h=True, convert_to_charmm22=True):
        if savepath[-4:] != '.pdb':
            throw_error('Output file must have ".pdb" extension')
        
        pdb = self.pick_template(mhcseq, pepseq)
        print('TEMPLATE: %s' % pdb)
        
        self.create_model_from_template(savepath, mhcseq, pepseq, pdb, add_polar_h)
        
        outfiles = [savepath]
        if convert_to_charmm22:
            outfiles += utils.prepare_pdb22(savepath, savepath[:-4])
        
        return outfiles