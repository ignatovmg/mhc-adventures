
# coding: utf-8

# In[1]:


import glob
import os
import pandas as pd
import sys


# In[3]:


from utils import merge_two
from wrappers import remove_files


# In[4]:


import define

if define.PLIP_DIR not in sys.path:
    sys.path.append(define.PLIP_DIR)

from plip.modules.preparation import PDBComplex

import plip.modules.config
plip.modules.config.PEPTIDES = ['B']


# In[6]:


def plip_read_pdb_complex(path):
    mol = PDBComplex()
    mol.load_pdb(path)
    mol.analyze()
    
    assert(len(mol.interaction_sets) == 1)
    inter = mol.interaction_sets.values()[0]
    return inter

def plip_read_pdb_rec_lig(rec, lig, remove_tmp=True):
    tmp_name = str(os.getpid()) + '.pdb'
    merge_two(tmp_name, rec, lig, keep_chain=True, keep_residue_numbers=True)
    inter = plip_read_pdb_complex(tmp_name)
    
    if remove_tmp:
        remove_files([tmp_name])
    
    return inter


# In[9]:


def _extract_keys(bond, keys):
    d = bond.__dict__
    return {k:d[k] for k in keys}

def _process_atom_list(atoms):
    return [(a.type, a.coords) for a in atoms]

def _process_common_bond(bond):
    keys = ['restype', 'resnr', 'reschain', 'restype_l', 'resnr_l', 'reschain_l']
    return _extract_keys(bond, keys)

def _process_bond(bond):
    rec = _process_common_bond(bond)
    btype = type(bond).__name__
    rec['bond'] = btype
    
    if btype == 'hydroph_interaction':
        rec['rec_atom_id'] = [bond.bsatom_orig_idx]
        rec['lig_atom_id'] = [bond.ligatom_orig_idx]
        rec['rec_coords'] = tuple(bond.bsatom.coords)
        rec['lig_coords'] = tuple(bond.ligatom.coords)
        rec['distance'] = bond.distance
        rec['type'] = ''
        
        add_info = {}
        add_info['rec_all_coords'] = _process_atom_list([bond.bsatom])
        add_info['lig_all_coords'] = _process_atom_list([bond.ligatom])
        rec['add_info'] = add_info
        
    elif btype == 'hbond':
        if bond.protisdon:
            rec['rec_atom_id'] = [bond.d_orig_idx]
            rec['lig_atom_id'] = [bond.a_orig_idx]
            rec['rec_coords'] = tuple(bond.d.coords)
            rec['lig_coords'] = tuple(bond.a.coords)
            
        else:
            rec['rec_atom_id'] = [bond.a_orig_idx]
            rec['lig_atom_id'] = [bond.d_orig_idx]
            rec['rec_coords'] = tuple(bond.a.coords)
            rec['lig_coords'] = tuple(bond.d.coords)
        rec['distance'] = bond.distance_ad
        rec['type'] = bond.type
        
        add_info = {'h_coords': bond.h.coords}
        add_keys = ['distance_ad', 'angle', 'protisdon', 'sidechain', 'atype', 'dtype']
        add_info.update(_extract_keys(bond, add_keys))
        
        if bond.protisdon:
            add_info['rec_all_coords'] = _process_atom_list([bond.d])
            add_info['lig_all_coords'] = _process_atom_list([bond.a])
        else:
            add_info['rec_all_coords'] = _process_atom_list([bond.a])
            add_info['lig_all_coords'] = _process_atom_list([bond.d])
        
        rec['add_info'] = add_info
        
    elif btype == 'saltbridge':
        if bond.protispos:
            prt = bond.positive
            lig = bond.negative
        else:
            lig = bond.positive
            prt = bond.negative
            
        rec['rec_atom_id'] = prt.atoms_orig_idx
        rec['lig_atom_id'] = lig.atoms_orig_idx
        rec['rec_coords'] = tuple(prt.center)
        rec['lig_coords'] = tuple(lig.center)
        rec['distance'] = bond.distance
        rec['type'] = ''
        
        add_info = _extract_keys(bond, ['protispos'])
        add_info['rec_all_coords'] = _process_atom_list(prt.atoms)
        add_info['lig_all_coords'] = _process_atom_list(lig.atoms)
        add_info['fgroup'] = lig.fgroup
        rec['add_info'] = add_info
        
    elif btype == 'pistack':
        rec['rec_atom_id'] = bond.proteinring.atoms_orig_idx
        rec['lig_atom_id'] = bond.ligandring.atoms_orig_idx
        rec['rec_coords'] = tuple(bond.proteinring.center)
        rec['lig_coords'] = tuple(bond.ligandring.center)
        rec['distance'] = bond.distance
        rec['type'] = bond.type
        
        add_info = _extract_keys(bond, ['angle', 'offset'])
        add_info['rec_all_coords'] = _process_atom_list(bond.proteinring.atoms)
        add_info['rec_ring_normal'] = bond.proteinring.normal
        add_info['rec_ring_type'] = bond.proteinring.type
        add_info['lig_all_coords'] = _process_atom_list(bond.ligandring.atoms)
        add_info['lig_ring_normal'] = bond.ligandring.normal
        add_info['lig_ring_type'] = bond.ligandring.type
        rec['add_info'] = add_info
        
    return rec


# In[10]:


def make_interaction_table(inter):
    table = {}
    for i, bond in enumerate(inter.all_itypes):
        table[i] = _process_bond(bond)
        
    table = pd.DataFrame(table).transpose()   
    return table


# In[11]:


def describe_interaction(inter):
    props = ['molweight', 'num_rot_bonds', 'num_rings', 'num_hbd', 'num_hba']
    d = _extract_keys(inter.ligand, props)
    d['num_unpaired_hba'] = inter.num_unpaired_hba
    d['num_unpaired_hbd'] = inter.num_unpaired_hbd
    d['bsite_orig'] = inter.bindingsite.bs_res
    d['contacts_orig'] = inter.interacting_res
    return d
