"""Stream-audit every sealed Top1 outcome against official source."""
import csv,gzip,json
from pathlib import Path

def main():
 data=json.load(gzip.open('experiments/auzo_official_history.json.gz','rt'));idx={r['draw_term']:i for i,r in enumerate(data)}
 count=0;current=None;seen=set();terms=set()
 with gzip.open('experiments/bingo_auzo.csv.gz','rt') as stream:
  for row in csv.DictReader(stream):
   term=row['draw_id'];i=idx[term]
   if term!=current:
    if current is not None:assert len(seen)==282
    assert term not in terms;terms.add(term);current=term;seen=set();actual=set(data[i]['numbers'])
   identity=(row['method'],row['window']);assert identity not in seen;seen.add(identity)
   assert row['source_term']==data[i-1]['draw_term']
   assert set(json.loads(row['actual_numbers']))==actual
   number=int(row['predicted_number']);assert 1<=number<=80
   assert int(row['hit'])==int(number in actual)
   count+=1
 assert len(seen)==282 and len(terms)==10000 and count==2820000
 report=dict(rows=count,targets=len(terms),candidates_per_target=282,duplicate_predictions=0,outcome_mismatches=0,passed=True)
 Path('reports/auzo_artifact_validation.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
