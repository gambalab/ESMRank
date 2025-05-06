import edlib
import numpy as np


def mask(aln_str, aln_pattern):
    return ''.join([s if p == '|' else '<mask>'
                    for s, p in zip(aln_str, aln_pattern)
                    if not (p == '-' and s == '-')])


def align_and_mask(wt_seq, mu_seq):
    aln = edlib.align(wt_seq, mu_seq, task='path')
    try:
        q, m, t = edlib.getNiceAlignment(aln, wt_seq, mu_seq).values()
    except:
        return '-', '-', np.nan
    return mask(q, m), mask(t, m), aln['editDistance']



