# MHC-adventures
Some tools for working with MHC structures

 How to
---
1. Install [Chimera](https://www.cgl.ucsf.edu/chimera/download.html) and Brikard 
and specify their location in `vars.json.tpl`
2. Run `./bootstrap.sh`
3. `source activate.sh`
4. `pytest -v`

So far usable and covered with tests are:
 * `from mhc_tools.sampling.generate_peptides import PeptideSampler`
 * `from mhc_tools.sampling.template_modelling import TemplateModeller`
 * `from mhc_tools.mhc_peptide import BasePDB`
 
The rest is in development