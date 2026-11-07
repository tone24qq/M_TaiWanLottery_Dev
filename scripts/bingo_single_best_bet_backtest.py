"""Independent causal Top1 experiment. No production imports or writes."""
import argparse,csv,gzip,json,hashlib,math
from pathlib import Path
import numpy as np
from scipy.stats import norm,binomtest
WINDOWS=(5,10,20,50,100,200,500)
COVERAGES=(.5,.25,.2,.1,.05,.01)

def interval(h,n,alpha=.05):
 if not n:return (0.,1.)
 z=norm.isf(alpha/2);p=h/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return float(c-w),float(c+w)

def features(x,t,w):
 past=x[max(0,t-w):t];long=x[max(0,t-500):t].mean(0);freq=past.mean(0)
 recent=x[max(0,t-500):t][::-1]
 gap=np.where(recent.any(0),recent.argmax(0),len(recent)).astype(float)
 cols=[freq,long,freq-long,gap/500,x[t-1]]
 for n in (1,3,5,10,20):cols.append(x[max(0,t-n):t].mean(0))
 for k in (1,2,3):
  a=np.zeros(80);a[k:]+=x[t-1,:-k];a[:-k]+=x[t-1,k:];cols.append(a/2)
 ids=np.arange(1,81);cols.extend([ids%2,ids>40,ids//10/8,ids%10/9])
 co=past.T@past/max(1,len(past));cols.append(co[:,x[t-1].astype(bool)].mean(1))
 return np.asarray(cols).T

def run(data,count=1000):
 if len(data)<count+100:raise ValueError('Insufficient causal warmup')
 from taiwan_lottery.ml.bingo_neighbor_evaluation import _history
 data=_history(data)
 x=np.zeros((len(data),80))
 for i,r in enumerate(data):x[i,np.array(r['numbers'])-1]=1
 rows=[];states={};history={};adaptive=[]
 # Online logistic models update only after each target is sealed and revealed.
 for t in range(20,len(data)):
  decisions=[]
  aw=WINDOWS[int(np.argmax([np.mean([v for _,v in adaptive[-100:] if _==w]) if any(a==w for a,b in adaptive[-100:]) else .25 for w in WINDOWS]))] if adaptive else 50
  for window in (*WINDOWS,'adaptive'):
   w=aw if window=='adaptive' else window;f=features(x,t,w);f=np.c_[np.ones(80),f]
   for kind in ('frequency','global','per_number','hybrid'):
    key=(str(window),kind)
    if key not in states:states[key]=(np.zeros(f.shape[1]),np.zeros_like(f))
    g,p=states[key];sg=1/(1+np.exp(-np.clip(f@g,-20,20)));sp=1/(1+np.exp(-np.clip((f*p).sum(1),-20,20)))
    s=(x[max(0,t-w):t].sum(0)+2)/(min(t,w)+8) if kind=='frequency' else sg if kind=='global' else sp if kind=='per_number' else (sg+sp)/2
    order=np.lexsort((np.arange(80),-s));top=int(order[0]);conf=float(s[top]);margin=float(s[top]-s[order[1]])
    previous=history.setdefault(key,[]);thresholds={str(c):float(np.quantile([a for a,b in previous[-500:]],1-c)) if len(previous)>=100 else None for c in COVERAGES}
    decisions.append((key,f,sg,sp,top,conf,margin,order,thresholds,w))
  # Target access occurs only after every candidate and threshold is frozen.
  actual=x[t]
  for key,f,sg,sp,top,conf,margin,order,thresholds,w in decisions:
   hit=int(actual[top]);g,p=states[key];g+=.02*(f.T@(actual-sg)/80-.001*g);p+=.02*((actual-sp)[:,None]*f-.001*p)
   history[key].append((conf,hit))
   if key[1]=='frequency' and key[0]!='adaptive':adaptive.append((int(key[0]),hit))
   if t>=len(data)-count:
    rows.append(dict(draw_id=data[t]['draw_term'],predicted_number=top+1,confidence=conf,actual_numbers=json.dumps(data[t]['numbers']),hit=hit,rank_of_best_hit=int(np.flatnonzero(actual[order])[0])+1,model_name=key[1],training_window=key[0],effective_window=w,feature_set='causal-local-v1',training_cutoff=data[t-1]['draw_term'],top2=int(order[1])+1,top3=int(order[2])+1,top1_score=conf,top2_score=float((conf-margin)),margin=margin,selective=json.dumps({c:thresholds[c] is not None and conf>thresholds[c] for c in thresholds})))
 return rows

def report(rows,n):
 lines=['每期只選一個號碼：尚未證明穩定高於 Random 25%。','', '分類 D：目前沒有通過獨立 fresh 驗證的優勢證據。此結果不是證明所有方法無效。','',f'現有資料1276期；實跑最近{n}期，32個預宣告候選各期各自只選一號。沒有選出或部署正式模型。5000／10000期及兩批1000～5000期fresh資料不足；現有快照曾供既有研究使用，不能稱為未參與開發的fresh test。','', '限制：global／per-number／hybrid 的 window 為特徵視窗，線上權重累積過去資料，並非每折只用前N期重訓；long-term特徵固定最多500期。因此本版未完成嚴格訓練視窗消融。', '', 'confidence 是未校正研究分數，不是已校準中獎機率。所有選擇門檻只取過去最多500期分數；嚴格大於門檻，平手會降低coverage。adaptive僅根據已揭露過去frequency結果選window。','', '| Model/window | Blind draws | Answered | Coverage | Hits | Top1 rate | Random | Absolute lift pp | Relative lift | 95% CI |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
 groups={}
 for r in rows:groups.setdefault((r['model_name'],r['training_window']),[]).append(r)
 details={}
 for key,rr in groups.items():
  h=sum(r['hit'] for r in rr);lo,hi=interval(h,len(rr));rate=h/len(rr)
  lines.append(f'| {key[0]}/{key[1]} | {n} | {len(rr)} | 100% | {h} | {rate:.2%} | 25% | {(rate-.25)*100:+.2f} | {rate/.25:.3f}× | {lo:.2%}–{hi:.2%} |')
  detail={'adjusted_p':min(1,binomtest(h,len(rr),.25,alternative='greater').pvalue*32),'simultaneous_ci':interval(h,len(rr),.05/32),'selective':{},'periods':{},'numbers':[],'confidence_deciles':[],'margin_deciles':[]}
  lines.extend(['',f'### {key[0]}/{key[1]} selective（過去門檻，非事後挑選）','', '| Target coverage | Answered | Actual coverage | Hits | Hit rate | 95% CI |','|---|---:|---:|---:|---:|---|'])
  for c in COVERAGES:
   subset=[r for r in rr if json.loads(r['selective'])[str(c)]];h2=sum(r['hit'] for r in subset);m=len(subset);ci=interval(h2,m)
   detail['selective'][str(c)]={'answered':m,'coverage':m/n,'hits':h2,'rate':h2/m if m else None,'ci':ci,'adjusted_p':min(1,binomtest(h2,m,.25,alternative='greater').pvalue*32*6) if m else 1}
   lines.append(f'| {c:.0%} | {m} | {m/n:.2%} | {h2} | {h2/m:.2%} | {ci[0]:.2%}–{ci[1]:.2%} |' if m else f'| {c:.0%} | 0 | 0% | 0 | 無樣本 | 0–100% |')
  for size in (250,500,1000):detail['periods'][str(size)]=[{'from':b[0]['draw_id'],'to':b[-1]['draw_id'],'samples':len(b),'rate':np.mean([r['hit'] for r in b]),'lift':np.mean([r['hit'] for r in b])-.25} for b in [rr[i:i+size] for i in range(0,len(rr),size)]]
  for number in range(1,81):
   sub=[r for r in rr if r['predicted_number']==number];hits=sum(r['hit'] for r in sub);detail['numbers'].append(dict(number=number,predictions=len(sub),hits=hits,rate=hits/len(sub) if sub else None,baseline=.25,lift=hits/len(sub)-.25 if sub else None))
  for field in ('confidence','margin'):
   for b in np.array_split(sorted(rr,key=lambda r:r[field]),10):detail[field+'_deciles'].append({'samples':len(b),'hits':sum(r['hit'] for r in b),'rate':sum(r['hit'] for r in b)/len(b)})
  details['/'.join(key)]=detail
 lines+=['','完整分段、全部80號及confidence／margin分桶見同名 JSON。分桶屬事後描述，不可作已驗證下注門檻；多重候選與subset均提供Bonferroni p值，未做fresh確認不宣稱成功。']
 return '\n'.join(lines),details

def main():
 a=argparse.ArgumentParser();a.add_argument('--source',default='docs/validation/official-draws-20261103.json.gz');a.add_argument('--count',type=int,default=1000);args=a.parse_args()
 data=json.load(gzip.open(args.source,'rt'));rows=run(data,args.count);again=run(data,args.count);assert rows==again
 Path('experiments').mkdir(exist_ok=True);Path('reports').mkdir(exist_ok=True)
 with open('experiments/bingo_single_best_bet.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 text,details=report(rows,args.count);Path('reports/bingo_single_best_bet.md').write_text(text);Path('reports/bingo_single_best_bet.json').write_text(json.dumps(details,indent=2))
 print(text.splitlines()[0]);print('rows',len(rows),'deterministic replay passed')
if __name__=='__main__':main()
