"""Auzo-inspired, explicitly defined causal statistics, not an exact site clone.

Input is an exclusive chronological prefix. Observed gaps are censored at the
left boundary; gap means use only complete inter-arrival intervals. Conditional
rates expose counts and use Beta(2,6) smoothing for ranking only.
"""
import numpy as np
WINDOWS=(3,5,10,20,30,50,100,200)

def auzo_features(history, *, detail=True):
 x=np.asarray(history,dtype=float)
 if x.ndim!=2 or x.shape[1]!=80 or not len(x):raise ValueError('Need nonempty prefix x80')
 if not np.isin(x,[0,1]).all() or not (x.sum(1)==20).all():raise ValueError('Each draw requires20 distinct numbers')
 n=len(x);ids=np.arange(80);rev=x[::-1];gap=np.where(rev.any(0),rev.argmax(0),n)
 result={'frequency':x.mean(0),'missing':gap.astype(float),'last_hit':x[-1], 'censored_gap':~x.any(0)}
 for w in WINDOWS:result['count_'+str(w)]=x[-w:].sum(0)
 for lag in (1,2,3):result['previous_'+str(lag)]=x[-lag] if n>=lag else np.zeros(80)
 if detail:
  means=[];medians=[];maxima=[];sequences=[]
  for j in ids:
   complete=np.diff(np.flatnonzero(x[:,j]))-1
   means.append(float(complete.mean()) if len(complete) else np.nan);medians.append(float(np.median(complete)) if len(complete) else np.nan)
   maxima.append(float(max(complete.max(initial=0),gap[j])));sequences.append(complete[-10:].tolist())
  result.update(average_gap=np.array(means),median_gap=np.array(medians),max_gap=np.array(maxima),gap_sequences=sequences)
 misses=rev==0
 streak=np.where(misses.any(0),misses.argmax(0),n)
 result['streak']=streak
 for lag in range(1,11):
  before=x[:-lag];after=x[lag:];trials=before.sum(0);hits=(before*after).sum(0)
  result['lag_'+str(lag)]=(hits+2)/(trials+8)
  result['lag_'+str(lag)+'_trials']=trials;result['lag_'+str(lag)+'_hits']=hits
  result['lag_score_'+str(lag)]=np.where(n>=lag,x[-min(lag,n)]*result['lag_'+str(lag)]+(1-x[-min(lag,n)])*.25,.25)
 result['repeat']=result['lag_score_1']
 before=x[:-1];after=x[1:];counts=before.T@after;trials=before.sum(0)
 result['drag']=((counts+2)/(trials[:,None]+8))[x[-1].astype(bool)].mean(0) if len(x)>1 else np.full(80,.25)
 last=np.maximum.accumulate(np.where(x>0,np.arange(n)[:,None],-1),axis=0)
 gaps=np.arange(n)[:,None]-last
 for k in range(11):
  mask=(gaps[:-1]==k)&(last[:-1]>=0);trials=mask.sum(0);hits=(mask*x[1:]).sum(0)
  result['miss_'+str(k)]=(hits+2)/(trials+8);result['miss_'+str(k)+'_trials']=trials;result['miss_'+str(k)+'_hits']=hits
 result['gap_hazard']=np.array([result['miss_'+str(min(int(gap[j]),10))][j] if gap[j]<=10 else .25 for j in ids])
 for distance in (1,2,3):
  numerator=np.zeros(80);denominator=np.zeros(80)
  for shift in (-distance,distance):
   sources=np.arange(max(0,-shift),min(80,80-shift));targets=sources+shift
   before=x[:-1,sources];after=x[1:,targets];trials=before.sum(0);hits=(before*after).sum(0)
   active=x[-1,sources];numerator[targets]+=active*(hits+2)/(trials+8);denominator[targets]+=active
   result['neighbor_'+str(shift)+'_trials']=trials;result['neighbor_'+str(shift)+'_hits']=hits
  result['neighbor_'+str(distance)]=np.divide(numerator,denominator,out=np.full(80,.25),where=denominator>0)
 for name,groups in [('segment',ids//10),('tail',(ids+1)%10),('size',ids>=40),('parity',(ids+1)%2)]:
  groups=groups.astype(int)
  means=np.bincount(groups,weights=x.mean(0))/np.bincount(groups)
  result[name]=means[groups]
 result['cooccurrence']=(x.T@x)[:,x[-1].astype(bool)].mean(1)/n
 for length in (2,3,4):
  counts=np.lib.stride_tricks.sliding_window_view(x,length,axis=1).prod(2).mean(0)
  scores=np.zeros(80)
  for offset in range(length):scores[offset:offset+len(counts)]+=counts
  result['consecutive_'+str(length)]=scores
 for name in ('segment','tail','size','parity'):
  groups=ids//10 if name=='segment' else (ids+1)%10 if name=='tail' else ids>=40 if name=='size' else (ids+1)%2
  best=int(np.argmax([result[name][np.flatnonzero(groups==g)[0]] for g in np.unique(groups)]));chosen=np.unique(groups)[best]
  result['filter_'+name]=np.where(groups==chosen,result['frequency'],-1)
 s=ids//10;t=(ids+1)%10
 result['filter_segment_tail']=np.where((result['filter_segment']>=0)&(result['filter_tail']>=0),result['frequency'],-1)
 if (result['filter_segment_tail']<0).all():result['filter_segment_tail']=result['filter_segment'].copy()
 return result
