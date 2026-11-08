import asyncio,gzip,json,sys
from datetime import date,timedelta
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from taiwan_lottery.scraper.parsers.bingo_parser import BingoScraper
async def main():
 sem=asyncio.Semaphore(3)
 async def day(d):
  async with sem:
   r=await BingoScraper(attempts=2,request_timeout=20)._fetch_day(d);print(d,len(r),flush=True);return r
 start=date(2026,9,1);end=date(2026,11,7)
 rows=await asyncio.gather(*(day(start+timedelta(days=i)) for i in range((end-start).days+1)))
 data=sorted([r for day in rows for r in day],key=lambda r:int(r['draw_term']))
 Path('experiments').mkdir(exist_ok=True)
 Path('experiments/auzo_official_history.json.gz').write_bytes(gzip.compress(json.dumps(data,default=str).encode(),mtime=0))
 print('saved',len(data),flush=True)
if __name__=='__main__':asyncio.run(main())
