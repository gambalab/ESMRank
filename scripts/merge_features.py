import pandas as pd
import os

sample_name = os.environ['SAMPLE_NAME']
esm_features = f'output/{sample_name}/esm_features.csv'
manual_features = f'output/{sample_name}/manual_features.csv'
out_file = f'output/{sample_name}/dataset.csv'


#%%

X = pd.read_csv(esm_features, index_col='ids')
Y = pd.read_csv(manual_features, index_col=0)
#%%
if 'Unnamed: 0' in X.columns:
    del X['Unnamed: 0']


Z =X.join(Y)

#%%

Z['esm_score'] = Z.wt_score - Z.mu_score

del Z['wt_score'],Z['mu_score']

#%%
Z.dropna(subset=['embedding_distance', 'structural_distance', 'Residue Weight', 'pKa1',
                                   'pl4', 'P1', 'NCISC', 'wt_len', 'rel_pos', 'distance', 'aggregation',
                                   'dist_to_domain', 'entropy', 'instability', 'flexibility', 'gravy',
                                   'isoelectric_point', 'tm_idx','esm_score']).to_csv(out_file)
print('done')
