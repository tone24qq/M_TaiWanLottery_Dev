"""Independent experimental ablation, no production state or network access."""
import csv,gzip,json,hashlib,bisect,math
from pathlib import Path
from taiwan_lottery.ml.bingo_models import ModelSuite,latest_features
import numpy as np
from scipy.stats import binomtest
from taiwan_lottery.ml.auzo_features import auzo_features,WINDOWS
from taiwan_lottery.ml.bingo_neighbor_evaluation import _history
from scripts.bingo_single_best_bet_backtest import interval
METHODS=('frequency','missing','gap_hazard','repeat','drag',*[f'lag_score_{i}' for i in range(1,11)],*[f'neighbor_{i}' for i in (1,2,3)],'segment','tail','size','parity','cooccurrence',*[f'consecutive_{i}' for i in (2,3,4)],'filter_segment','filter_tail','filter_segment_tail','filter_size','filter_parity')

def replay(data,count=1000):
 data=_history(data)
 if len(data)<count+200:raise ValueError('Need count+200 verified draws')
 x=np.zeros((len(data),80))
 for i,r in enumerate(data):x[i,np.array(r['numbers'])-1]=1
 rows=[];past={};ordered={};families={}
 for index in range(len(data)-count,len(data)):
  decisions=[]
  day=str(data[index]['draw_datetime'])[:10]
  same=[i for i in range(index) if str(data[i]['draw_datetime'])[:10]==day]
  contexts=[(str(w),x[max(0,index-w):index]) for w in WINDOWS]+[('today',x[same])]
  for window,context in contexts:
   f=auzo_features(context,detail=False) if len(context) else {m:np.full(80,.25) for m in METHODS}
   for method in METHODS:
    scores=f[method];order=np.lexsort((np.arange(80),-scores));top=int(order[0]);score=float(scores[top]);key=(method,window);old=past.setdefault(key,[])
    ranked=ordered.setdefault(key,[])
    def quantile(c):
     pos=(len(ranked)-1)*(1-c);lo=math.floor(pos);hi=math.ceil(pos)
     return ranked[lo]+(ranked[hi]-ranked[lo])*(pos-lo)
    thresholds={str(c):quantile(c) if len(ranked)>=100 else None for c in (.5,.25,.1,.05,.01)}
    decisions.append((key,top,score,thresholds))
  matrix=x[max(0,index-1000):index];oldrow=latest_features(matrix)
  for old in ('frequency','cdm'):
   scores=ModelSuite._score_one(None,old,None,matrix,oldrow);top=int(np.argmax(scores))
   decisions.append((('old_'+old,'1000'),top,float(scores[top]),{str(c):None for c in (.5,.25,.1,.05,.01)}))
  seed=int.from_bytes(hashlib.sha256(data[index]['draw_term'].encode()).digest()[:8],'big')
  decisions.append((('random','none'),int(np.random.default_rng(seed).integers(80)),.25,{str(c):None for c in (.5,.25,.1,.05,.01)}))
  # Every method/window is frozen before any answer is read.
  actual=set(data[index]['numbers'])
  for (method,window),top,score,thresholds in decisions:
   key=(method,window);values=past.setdefault(key,[]);ranked=ordered.setdefault(key,[])
   if len(values)>=500:ranked.pop(bisect.bisect_left(ranked,values[-500]))
   bisect.insort(ranked,score);values.append(score)
   rows.append(dict(draw_id=data[index]['draw_term'],source_term=data[index-1]['draw_term'],method=method,window=window,predicted_number=top+1,confidence=score,hit=int(top+1 in actual),actual_numbers=json.dumps(sorted(actual)),selective=json.dumps({c:v is not None and score>v for c,v in thresholds.items()})))
 return rows

def main():
 path=Path('experiments/auzo_official_history.json.gz');data=json.load(gzip.open(path,'rt'));rows=replay(data,10000);groups={}
 for count in (1000,5000,10000):
  terms={r['draw_term'] for r in data[-count:]}
  for r in rows:
   if r['draw_id'] in terms:groups.setdefault((r['method'],r['window'],count),[]).append(r)
 family=len(groups);summaries=[]
 for (m,w,count),r in groups.items():
  h=sum(x['hit'] for x in r);n=len(r);p=min(1.,binomtest(h,n,.25,alternative='greater').pvalue*family*6)
  halves=[sum(x['hit'] for x in r[i:i+500])/len(r[i:i+500]) for i in range(0,n,500)]
  subs={}
  for c in ('.5','.25','.1','.05','.01'):
   key=str(float(c));s=[x for x in r if json.loads(x['selective'])[key]];hh=sum(x['hit'] for x in s)
   subs[key]=dict(answered=len(s),coverage=len(s)/n,hits=hh,rate=hh/len(s) if s else None,ci=interval(hh,len(s)),adjusted_p=min(1.,family*6*10000*math.exp(-2*len(s)*max(0.,hh/len(s)-.25)**2)) if s else 1.)
  summaries.append(dict(method=m,window=w,predictions=n,hits=h,rate=h/n,lift=h/n-.25,relative_lift=h/n/.25,ci=interval(h,n),simultaneous_ci=interval(h,n,.05/(family*6)),adjusted_p=p,period_500=halves,selective=subs,development_pass=bool(binomtest(sum(x['hit'] for x in r[:500]),500,.25,alternative='greater').pvalue*family<.05)))
 summaries.sort(key=lambda s:(-s['rate'],s['method'],s['window']))
 Path('reports').mkdir(exist_ok=True);Path('experiments').mkdir(exist_ok=True)
 with gzip.open('experiments/bingo_auzo.csv.gz','wt') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 result=dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),data_draws=len(data),blind_draws=[1000,5000,10000],family=family,rows=len(rows),rankings=summaries,combined_eligible=[s for s in summaries if s['development_pass']],fresh_confirmation=False,production_changed=False)
 Path('reports/bingo_auzo.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 lines=['# 奧索型 Top1 實驗','', '目前無法證明奧索型分析穩定高於25%；不是對所有奧索算法的完整否定。','',f'官方資料{len(data)}期，已實跑最近1000、5000、10000期；{family}個方法/視窗/樣本大小比較，逐期紀錄{len(rows)}筆。沒有足夠真正未參與研究的fresh確認。','', '網站部分內容無法存取，以下為明確定義的奧索型研究延伸，未聲稱精確複製未公開算法。所有分數只用<t資料；分數不是校準概率。未連接production。','', '| 分析方法 | Window | Predictions | Hits | Top1命中率 | Lift pp | Relative | 95% CI | 校正p | 每500期 |','|---|---:|---:|---:|---:|---:|---:|---|---:|---|']
 for s in summaries:lines.append(f"| {s['method']} | {s['window']} | {s['predictions']} | {s['hits']} | {s['rate']:.2%} | {s['lift']*100:+.2f} | {s['relative_lift']:.3f} | {s['ci'][0]:.2%}–{s['ci'][1]:.2%} | {s['adjusted_p']:.4f} | {s['period_500']} |")
 lines+=['','## Evidence gate','',f"前500期多重校正通過候選：{len(result['combined_eligible'])}。沒有合格單項時不擬合combined/logistic/LightGBM/ranking，避免違反先證據後組合。前500期僅作探索門檻，其後資料也不冒稱fresh；本次未獲任何正式選模授權。",'', '完整五種過去信心門檻的coverage、answered、hits、rate與區間見JSON；平手嚴格不回答，實際coverage可低於目標。排名為探索性結果，不因排名第一宣稱有效。']
 Path('reports/bingo_auzo.md').write_text('\n'.join(lines))
 from scripts.auzo_conditional_evidence import main as evidence
 evidence()
 from scripts.finalize_auzo_report import finalize
 finalize()
 print(json.dumps({k:v for k,v in result.items() if k not in ('rankings','combined_eligible')},ensure_ascii=False));print('development eligible',len(result['combined_eligible']))
if __name__=='__main__':main()
