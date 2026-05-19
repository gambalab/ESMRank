import argparse
import os

import pandas as pd

sample_name = os.environ['SAMPLE_NAME']
parser = argparse.ArgumentParser(description='generate variants from fasta')
parser.add_argument('--input','-i', type=str,help='input fasta file path')
parser.add_argument('-sub', action='store_true')
parser.add_argument('-indel', action='store_true')
parser.add_argument('-csv', action='store_true')
parser.add_argument('-alanines', action='store_true')
args = parser.parse_args()


assert ( args.csv ^ ( args.sub or args.indel or args.alanines  ) )
if args.csv:
    pd.read_csv(args.input).to_csv(f'output/{sample_name}/temp.csv',index=None)
    exit()

with open( args.input  ,'r') as f:
    seq = ''.join(f.read().split('\n')[1:])

aa = 'ACDEFGHIKLMNPQRSTVWY*'
res = {}

if args.alanines:
    assert not (args.csv or args.indel or args.sub)
    res.update({
        f'{seq[i]}{i+1}A' if seq[i] != 'A' else f'A{i+1}G':{

            'seq_wt':seq,
            'seq_mu': seq[:i] + ( 'A' if seq[i]!='A' else 'G') + seq[i+1:]

        }
        for i in range(len(seq))
    })


if args.sub:
    assert not ( args.alanines or args.csv )
    res.update( # generate substitutions
    {
        f'{seq[i]}{i+1}{j}':{

            'seq_wt':seq,
            'seq_mu':seq[:i] + j + seq[i+1:]

        }
        for i in range(len(seq))
        for j in aa
        if not ( j=='*' and i==0)
    })

if args.indel:
    assert not ( args.alanines or args.csv )
    res.update(  # generate deletions
    {
        f'{seq[i]}{i+1}DEL':{

            'seq_wt':seq,
            'seq_mu': seq[:i] + seq[i+1:]
        }
        for i in range(len(seq))
    })
    res.update(  # generate insertion
    {
        f'{i+1}INS{j}':
        {
            'seq_wt':seq,
            'seq_mu':seq[:i]+ j + seq[i:] 
        }
        for i in range(len(seq))
        for j in aa[:-1]
    })


res = pd.DataFrame.from_dict(res,orient='index')
res.index.name = 'hgvsp'
res.to_csv(f'output/{sample_name}/temp.csv')
