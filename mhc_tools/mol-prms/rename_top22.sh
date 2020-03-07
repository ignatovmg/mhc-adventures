#!/usr/bin/env bash
sed 's/HN/H /g' top_all22_prot.rtf > top_all22_prot_changed_atom_names.rtf

for R in HSD HSE HSP CYS GLN SER TRP PRO LYS PHE MET ASN LEU ARG ASP GLU TYR; 
    do sed -i "/RESI $R/,/RESI/ s/HB1/HB3/g" top_all22_prot_changed_atom_names.rtf
done

for R in GLN PRO LYS MET ARG GLU;
    do sed -i "/RESI $R/,/RESI/ s/HG1/HG3/g" top_all22_prot_changed_atom_names.rtf
done

sed -i "/RESI SER/,/RESI/ s/HG1/HG/g" top_all22_prot_changed_atom_names.rtf
sed -i "/RESI CYS/,/RESI/ s/HG1/HG/g" top_all22_prot_changed_atom_names.rtf
sed -i "/RESI GLY/,/RESI/ s/HA1/HA3/g" top_all22_prot_changed_atom_names.rtf
sed -i "/RESI PRO/,/RESI/ s/HD1/HD3/g" top_all22_prot_changed_atom_names.rtf
sed -i "/RESI LYS/,/RESI/ s/HE1/HE3/g" top_all22_prot_changed_atom_names.rtf
sed -i "/RESI LYS/,/RESI/ s/HD1/HD3/g" top_all22_prot_changed_atom_names.rtf
sed -i "/RESI ILE/,/RESI/ s/HD1/HD11/g" top_all22_prot_changed_atom_names.rtf
sed -i "/RESI ILE/,/RESI/ s/HD2/HD12/g" top_all22_prot_changed_atom_names.rtf
sed -i "/RESI ILE/,/RESI/ s/HD3/HD13/g" top_all22_prot_changed_atom_names.rtf
sed -i "/RESI ILE/,/RESI/ s/HG11/HG13/g" top_all22_prot_changed_atom_names.rtf
sed -i "/RESI ARG/,/RESI/ s/HD1/HD3/g" top_all22_prot_changed_atom_names.rtf
sed -i "/RESI ILE/,/RESI/ s/CD/CD1/g" top_all22_prot_changed_atom_names.rtf

# termini
sed -i -r "/PRES/,/END/ s/HT([1-3])/H\1/g" top_all22_prot_changed_atom_names.rtf
sed -i "/PRES/,/END/ s/OT1/O/g" top_all22_prot_changed_atom_names.rtf
sed -i "/PRES/,/END/ s/OT2/OXT/g" top_all22_prot_changed_atom_names.rtf