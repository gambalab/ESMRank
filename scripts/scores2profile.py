import re
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='build protein profile by aggregating position-wise and smoothing')
parser.add_argument('--input','-i', type=str,help='input fasta file path')
parser.add_argument('-sub', action='store_true')
parser.add_argument('-indel', action='store_true')
parser.add_argument('-csv', action='store_true')
parser.add_argument('-alanines', action='store_true')
args = parser.parse_args()

sample_name = os.environ['SAMPLE_NAME']
in_file = f'output/{sample_name}/predictions.csv'
out_file = f'output/{sample_name}/profile.tsv'
window_size = 5
mode = 'alanines' if  args.alanines else 'sub' if args.sub else 'not supported'

if mode not in {'sub', 'alanines'}: print('To build a profile run in fasta mode with -csv or -alanines flags');exit()

data = pd.read_csv(in_file)

#%%
position_scores = None
data['pos'] = data['ids'].map(lambda x: int(re.findall(r'\d+', x)[0]))
if mode == 'sub':
    position_scores = data.groupby('pos').prediction.mean()
    position_scores = pd.DataFrame(position_scores).reset_index()
elif mode == 'alanines':
        position_scores = data[['pos','prediction']].copy()


position_scores['score_smooth'] = position_scores['prediction'].rolling(window=5, center=True, min_periods=1, win_type='gaussian').mean(std=1)
result = position_scores[['pos','score_smooth']].copy()
result.columns = ['pos','score']
result.to_csv(out_file, index=False,sep='\t')
result.plot(x = 'pos', y = 'score')
plt.savefig(out_file.replace('.tsv','.png'), dpi=300)
