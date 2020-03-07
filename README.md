# MHC-adventures
Some tools for working with MHC structures

 How to
---
1. Install [Chimera](https://www.cgl.ucsf.edu/chimera/download.html) and Brikard and specify their location in vars.json.tpl
2. Run `./bootstrap.sh`
3. `source activate.sh`
4. `pytest -v -s`

So far usable and covered with tests are:
 * `from mhc_adventures.sampling.generate_peptides import PeptideSampler`
 * `from mhc_adventures.sampling.template_modelling import TemplateModeller`
 * `from mhc_adventures.mhc_peptide import BasePDB`
 
The rest is in development