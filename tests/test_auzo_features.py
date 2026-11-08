import numpy as np
import pytest
from taiwan_lottery.ml.auzo_features import auzo_features

def sample():
 x=np.zeros((12,80));x[:,0:20]=1;x[3:6,0]=0;x[3:6,79]=1
 return x

def test_missing_gap_and_complete_intervals():
 f=auzo_features(sample());assert f['missing'][0]==0 and f['average_gap'][0]==3/8
 assert f['censored_gap'][30] and f['missing'][30]==12

def test_scalar_lag_counts():
 x=sample();f=auzo_features(x)
 for lag in range(1,11):
  assert f[f'lag_{lag}_hits'][0]==sum(x[t-lag,0]*x[t,0] for t in range(lag,len(x)))

def test_fast_and_detailed_scores_equal():
 a=auzo_features(sample());b=auzo_features(sample(),detail=False)
 for k in b:np.testing.assert_equal(a[k],b[k])

def test_no_neighbor_wrap():
 x=np.zeros((8,80));x[:,60:80]=1;f=auzo_features(x)
 assert f['neighbor_1'][0]==.25

def test_reject_invalid_draw():
 with pytest.raises(ValueError):auzo_features(np.zeros((10,80)))

def test_replay_freezes_top1_before_target_and_future_answers():
 import gzip,json,copy
 from scripts.bingo_auzo_backtest import replay
 with gzip.open('docs/validation/official-draws-20261103.json.gz','rt') as f:rows=json.load(f)[:205]
 first=replay(rows,count=2);changed=copy.deepcopy(rows)
 changed[-2]['numbers']=list(range(1,21));changed[-1]['numbers']=list(range(61,81))
 later=replay(changed,count=2);term=rows[-2]['draw_term']
 keys=('draw_id','source_term','method','window','predicted_number','confidence','selective')
 assert [{k:r[k] for k in keys} for r in first if r['draw_id']==term]==[{k:r[k] for k in keys} for r in later if r['draw_id']==term]
 assert first==replay(rows,count=2)
