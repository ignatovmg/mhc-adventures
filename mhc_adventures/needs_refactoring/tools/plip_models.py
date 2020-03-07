
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
import logging

np.random.seed(123456)

from ilovemhc.plip_wrapper import plip_read_pdb_complex, make_interaction_table, describe_interaction

table_path = '/gpfs/projects/KozakovGroup/mhc_learning/analysis/affinity_data/affinity_clean-15012019.csv'
#mhc_seq_path = '/gpfs/projects/KozakovGroup/mhc_learning/analysis/allele_sequences_reduced.csv'
nproc = 2

aff_table = pd.read_csv(table_path, low_memory=False)

#mhc_seq_table = pd.read_csv(mhc_seq_path, sep=' ')
#mhc_seq_dict = dict(zip(mhc_seq_table.allele_short, mhc_seq_table.sequence))

models_list = '/gpfs/projects/KozakovGroup/mhc_learning/analysis/affinity_data/models/models2/test_list'
with open(models_list, 'r') as f:
    models = [x.strip() for x in f]

aff_lines = [int(x.split('-')[1]) for x in models]
aff_table = aff_table.loc[aff_lines, :]
aff_table.to_csv('affinity.csv', index=False)

# In[ ]:
exceptions = []


def run(pid):
    detailed_all = []
    general_all = {}
    print(pid)

    for modeli, model in enumerate(models):
        if modeli % nproc != pid:
            continue
        
        print(modeli, model)
        
        try:
            #allele = row.allele
            #peptide = row.peptide
            #allele_seq = mhc_seq_dict[allele]

            #save_path = os.path.join(models_dir, pdb_name)
            #template, _, nmin, psf = modeller.create_model(save_path, allele_seq, peptide)
            #cur_resi_map = resi_map[template]

            inter = plip_read_pdb_complex(model)
            detailed = make_interaction_table(inter)

            general = describe_interaction(inter)
            general['affinity_line'] = modeli
            #general['bsite_renum'] = [cur_resi_map[int(x[:-1])] for x in general['bsite_orig']]
            #general['contacts_renum'] = [cur_resi_map[int(x[:-1])] for x in general['contacts_orig']]

            #struct_id = os.path.basename(nmin)
            detailed['pdb'] = os.path.basename(model)
            detailed['affinity_line'] = modeli

            #pdb_list.append(pdb_name)
            detailed_all.append(detailed)
            general_all[os.path.basename(model)] = general
        except Exception as e:
            logging.exception(e)
            exceptions.append(modeli)

    general_all = pd.DataFrame(general_all).transpose()
    detailed_all = pd.concat(detailed_all)
    detailed_all.reset_index(drop=True, inplace=True)
    
    return detailed_all, general_all


# In[ ]:


pool = Pool(nproc)
output = pool.map(run, range(nproc))
#output = map(run, range(nproc))


# In[ ]:


zipped = zip(*output)

#pdb_list = reduce(lambda x,y: x+y, zipped[0])

detailed_all = pd.concat(zipped[0]).reset_index(drop=True)

general_all = pd.concat(zipped[1]).reset_index()


# In[ ]:

detailed_all.to_csv('detailed_interaction.csv', index=False)

general_all.to_csv('general_interaction.csv', index=False)

print("Exceptions:")
print(exceptions)

#aff_small.loc[~aff_small.index.isin(exceptions), :].to_csv("affinity-toy.csv", index=False)
