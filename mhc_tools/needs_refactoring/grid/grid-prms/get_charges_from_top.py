import pandas as pd
from subprocess import check_output
from StringIO import StringIO

charges = check_output('sed -nr "s/^ATOM +\S+ +(\S+) +(\S+) +.*\$/\\1 \\2/p" ../mol-prms/top_all22_prot_changed_atom_names.rtf | sort -k1', shell=True)
#print charges
table = pd.read_csv(StringIO(charges), sep=' ', header=None)
table = table.groupby(0).describe()
table.to_csv("top22_charges.csv", float_format='%.3f')
