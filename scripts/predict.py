import pandas as pd
import xgboost as xgb
import os

sample_name=os.environ['SAMPLE_NAME']
print('loading model')
model =  xgb.XGBRanker()
model.load_model('model/esm_xgb_rank_rrf.json')

print('loading features')
data = pd.read_csv(f'output/{sample_name}/dataset.csv')

print('running predictions')
data['prediction'] = model.predict(data[model.feature_names_in_])

print('writing results')
data[['ids','prediction']].to_csv(f'output/{sample_name}/predictions.csv',index=None)

