from path import Path

terminal = [['H1', 'HC'], ['H2', 'HC'], ['H3', 'HC'], ['OXT', 'OC']]


def fmt_name(aname):
    return '%4s' % ('%-3s' % aname)


with open('top_all22_prot_changed_atom_names.rtf', 'r') as f:
    lines = []
    line = ''
    for line in f:
        if line.startswith('RESI'):
            resn = line.split()[1]
            lines += [' '.join([fmt_name(aname), resn, atype]) for aname, atype in terminal]
        if line.startswith('ATOM'):
            aname, atype = line.split()[1:3]
            aname = fmt_name(aname)
            lines.append(' '.join([aname, resn, atype]))
        if line.startswith('PRES'):
            break

lines += ['']
out = Path('types_new.csv')
out.write_lines(lines)
