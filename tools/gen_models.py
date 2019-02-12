
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys
import os
import subprocess as sps
import multiprocessing as mlp
from multiprocessing import Pool
import logging

np.random.seed(123456)

# In[ ]:


libpath = '/gpfs/projects/KozakovGroup/mhc_learning/mhc-adventures/ilovemhc'
if libpath not in sys.path:
    sys.path.append(libpath)


# In[ ]:


import template_modelling
reload(template_modelling)

import plip_wrapper
reload(plip_wrapper)
from plip_wrapper import plip_read_pdb_complex, make_interaction_table, describe_interaction


# In[ ]:


table_path = '/gpfs/projects/KozakovGroup/mhc_learning/analysis/affinity_data/affinity_clean-15012019.csv'
mhc_seq_path = '/gpfs/projects/KozakovGroup/mhc_learning/analysis/allele_sequences_reduced.csv'
models_dir = 'models'
nproc = 4


# In[ ]:


sps.check_output('mkdir -p %s' % models_dir, shell=True)


# In[ ]:


aff_table = pd.read_csv(table_path, low_memory=False)


# In[ ]:


aff_small = aff_table.sample(n=1000)


# In[ ]:


aff_small.reset_index(drop=True, inplace=True)


# In[ ]:


aff_small.head()


# In[ ]:


aff_small.to_csv("affinity-toy.csv", index=False)


# In[ ]:


mhc_seq_table = pd.read_csv(mhc_seq_path, sep=' ')


# In[ ]:


mhc_seq_dict = dict(zip(mhc_seq_table.allele_short, mhc_seq_table.sequence))


# In[ ]:


modeller = template_modelling.TemplateModeller()


# In[ ]:


resi_map = []
for i, row in modeller.pdb_table.iterrows():
    old = map(int, row.resi_orig.split(','))
    new = map(int, row.resi_aln.split(','))
    resi_map.append((row.pdb, dict(zip(old, new))))
    
resi_map = dict(resi_map)


# In[ ]:
exceptions = []

def run(pid):
    detailed_all = []
    general_all = {}
    pdb_list = []
    
    print(pid)
    
    counter = -1
    for rowi, row in aff_small.iterrows():
        counter += 1
        if counter % nproc != pid:
            continue
        
        print(row)
        
        try:
            pdb_name = '%06i.pdb' % rowi
            allele = row.allele
            peptide = row.peptide
            allele_seq = mhc_seq_dict[allele]

            save_path = os.path.join(models_dir, pdb_name)
            template, _, nmin, psf = modeller.create_model(save_path, allele_seq, peptide)
            cur_resi_map = resi_map[template]

            inter = plip_read_pdb_complex(nmin)
            detailed = make_interaction_table(inter)

            general = describe_interaction(inter)
            general['bsite_renum'] = [cur_resi_map[int(x[:-1])] for x in general['bsite_orig']]
            general['contacts_renum'] = [cur_resi_map[int(x[:-1])] for x in general['contacts_orig']]

            struct_id = os.path.basename(nmin)
            detailed['pdb'] = struct_id

            pdb_list.append(pdb_name)
            detailed_all.append(detailed)
            general_all[struct_id] = general
        except Exception as e:
            logging.exception(e)
            exceptions.append(rowi)

    general_all = pd.DataFrame(general_all).transpose()
    detailed_all = pd.concat(detailed_all)
    detailed_all.reset_index(drop=True, inplace=True)
    
    return pdb_list, detailed_all, general_all


# In[ ]:


pool = Pool(nproc)
output = pool.map(run, range(nproc))
#output = map(run, range(nproc))


# In[ ]:


zipped = zip(*output)

pdb_list = reduce(lambda x,y: x+y, zipped[0])

detailed_all = pd.concat(zipped[1]).reset_index(drop=True)

general_all = pd.concat(zipped[2]).reset_index()


# In[ ]:


with open('models.list', 'w') as f:
    f.write('\n'.join(pdb_list) + '\n')

detailed_all.to_csv('detailed_interaction.csv', index=False)

general_all.to_csv('general_interaction.csv', index=False)

print("Exceptions:")
print(exceptions)

aff_small.loc[~aff_small.index.isin(exceptions), :].to_csv("affinity-toy.csv", index=False)
