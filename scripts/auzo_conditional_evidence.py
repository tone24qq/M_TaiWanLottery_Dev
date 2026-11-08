"""Descriptive full counts plus independent-pair conditional evidence."""
import csv,gzip,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from scripts.bingo_single_best_bet_backtest import interval
from taiwan_lottery.ml.bingo_neighbor_evaluation import _history

def main():
 data=_history(json.load(gzip.open('experiments/auzo_official_history.json.gz','rt')))[-1000:]
 x=np.zeros((len(data),80),int)
 for i,r in enumerate(data):x[i,np.array(r['numbers'])-1]=1
 rows=[]
 for lag in range(1,11):
  pairs=np.arange(lag,len(x),lag+1)
  for source in range(80):
   targets=range(max(0,source-3),min(80,source+4)) if lag==1 else (source,)
   for target in targets:
    raw=x[:-lag,source];h=int((raw*x[lag:,target]).sum());n=int(raw.sum())
    mask=x[pairs-lag,source]>0;nn=int(mask.sum());hh=int(x[pairs[mask],target].sum())
    rows.append(dict(source=source+1,target=target+1,lag=lag,raw_triggers=n,raw_hits=h,raw_rate=h/n if n else None,inference_triggers=nn,inference_hits=hh,inference_rate=hh/nn if nn else None,baseline=.25,p=binomtest(hh,nn,.25,alternative='greater').pvalue if nn else 1.))
 family=len(rows)
 for r in rows:
  r['adjusted_p']=min(1.,r.pop('p')*family);r['ci_low'],r['ci_high']=interval(r['inference_hits'],r['inference_triggers'],.05/family);r['evidence']='exploratory; no fresh confirmation'
 with open('reports/auzo_conditional_evidence.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 print('conditional relations',len(rows),'family',family)
if __name__=='__main__':main()
