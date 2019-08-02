import pandas as pd

from . import define

_GDOMAINS_TABLE = pd.read_csv(define.TEMPLATE_MODELLER_DEFAULT_TABLE)

#_ALLELE_DICT =


class BaseChain(object):
    def __init__(self, seq):
        self.seq = seq

    def __len__(self):
        return len(self.seq)


class MHCI(BaseChain):
    def __init__(self):
        super(MHCI, self).__init__()


class Peptide(BaseChain):
    def __init__(self):
        super(Peptide, self).__init__()


class Complex(object):
    def __init__(self):
        super(Complex, self).__init__()

