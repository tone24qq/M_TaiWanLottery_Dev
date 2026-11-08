"""Generate the ranked summary and conservative selective evidence reproducibly."""
import json,math,hashlib,gzip
from pathlib import Path
from scipy.stats import binomtest
from scripts.bingo_single_best_bet_backtest import interval

def finalize():
 r=json.loads(Path('reports/bingo_auzo.json').read_text());family=r['family']
 data=json.load(gzip.open('experiments/auzo_official_history.json.gz','rt'))
 for s in r['rankings']:
  terms=data[-s['predictions']:]
  s['period_details']=[dict(start=terms[i*500]['draw_term'],end=terms[min((i+1)*500,len(terms))-1]['draw_term'],hit_rate=rate,lift=rate-.25) for i,rate in enumerate(s['period_500'])]
  s['adjusted_p']=min(1.,binomtest(s['hits'],s['predictions'],.25,alternative='greater').pvalue*family*6)
  s['simultaneous_ci']=interval(s['hits'],s['predictions'],.05/(family*6))
  for v in s['selective'].values():
   n=v['answered'];rate=v['rate'];budget=family*6*10000
   v['adjusted_p']=min(1.,budget*math.exp(-2*n*max(0.,rate-.25)**2)) if n else 1.
   width=math.sqrt(math.log(2*budget/.05)/(2*n)) if n else 1.
   v['sequential_simultaneous_ci']=[max(0.,rate-width),min(1.,rate+width)] if n else [0.,1.]
   v['inference_method']='Hoeffding union over N<=10000 and all comparisons/subsets; Wilson CI descriptive'
 Path('reports/bingo_auzo.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
 p=Path('reports/bingo_auzo.md');raw=''
 summary=['# E：本次已實作分析尚未證明高於Random 25%','', '官方來源13804期；實跑最近1000／5000／10000期，282候選，每期各自只選一號，合計2820000筆。多重比較涵蓋846種方法／視窗／樣本大小及整段與五種信心子集。','', '僅適用於本次已實作的方法，不能聲稱完整複製或否定所有奧索功能。原站存取與定義差異見 [巡覽紀錄](auzo_site_inventory.md)。','', '下表取10000期各方法最高觀察率的探索排名，視窗經事後排序；CI涵蓋846比較×整段及5種子集的Bonferroni校正（Wilson近似），排名不等於有效。','', '| 方法 | Window | Top1命中率 | Lift pp | 校正95% CI | 每500期高於25% |','|---|---:|---:|---:|---|---:|']
 seen=set()
 for s in r['rankings']:
  if s['predictions']!=10000 or s['method'] in seen:continue
  seen.add(s['method']);lo,hi=s['simultaneous_ci'];periods=s['period_500']
  summary.append(f"| {s['method']} | {s['window']} | {s['rate']:.2%} | {s['lift']*100:+.2f} | {lo:.2%}–{hi:.2%} | {sum(v>.25 for v in periods)}/{len(periods)} |")
 summary+=['','Production未變更。舊Frequency/CDM只作baseline，未重播全部舊八模型；沒有真正fresh確認。0個單項通過前段校正，故未訓練combined/logistic/LightGBM/ranking，不能宣稱已評估組合模型。','', '高信心回答數依過去分數適應，JSON提供對N≤10000與全部子集的Hoeffding聯合界；名目Wilson區間僅供描述。沒有子集通過校正。','', '重現：`python -m scripts.fetch_auzo_official_history`、`python -m scripts.bingo_auzo_backtest`、`python -m scripts.auzo_conditional_evidence`。CPU執行緒建議1。','',raw]
 summary=summary[:-1]
 summary+=['# 全部方法／視窗／樣本大小排名','', '| 方法 | Window | Predictions | Hits | Top1率 | Lift pp | Relative | 校正95%CI | 校正p | 每500期 |','|---|---:|---:|---:|---:|---:|---:|---|---:|---|']
 for s in r['rankings']:
  lo,hi=s['simultaneous_ci']
  summary.append(f"| {s['method']} | {s['window']} | {s['predictions']} | {s['hits']} | {s['rate']:.2%} | {s['lift']*100:+.2f} | {s['relative_lift']:.3f}× | {lo:.2%}–{hi:.2%} | {s['adjusted_p']:.4f} | {s['period_500']} |")
 p.write_text('\n'.join(summary))
 paths=['taiwan_lottery/ml/auzo_features.py','scripts/bingo_auzo_backtest.py','scripts/finalize_auzo_report.py','scripts/auzo_conditional_evidence.py','scripts/fetch_auzo_official_history.py','experiments/auzo_official_history.json.gz','experiments/bingo_auzo.csv.gz','reports/bingo_auzo.md','reports/bingo_auzo.json','reports/auzo_site_inventory.md','reports/auzo_conditional_evidence.csv']
 Path('reports/auzo_manifest.json').write_text(json.dumps(dict(files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},rows=r['rows'],counts=r['blind_draws'],source_draws=r['data_draws'],fresh=False,production_changed=False),indent=2))
if __name__=='__main__':finalize()
