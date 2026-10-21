#!/usr/bin/env python3
"""Catalog-only feasibility audit. Never fetch PDFs or feature outcomes."""
import csv,gzip,hashlib,json,os,pathlib,re,urllib.request,urllib.parse
ROOT=pathlib.Path(__file__).resolve().parents[3];OUT=ROOT/'results/review_round2/expansion'
DATA_DIR=pathlib.Path(os.environ.get('BEFORE_AFTER_AI_DATA_DIR', pathlib.Path.home()/'.cache'/'before-after-ai'))
CACHE=DATA_DIR/'review_round2_expansion_metadata'
SOURCES={
'cvpr2020':[f'https://openaccess.thecvf.com/CVPR2020.py?day=2020-06-{d}' for d in (16,17,18)],
'cvpr2025':['https://openaccess.thecvf.com/CVPR2025?day=all'],
'icml2020':['https://proceedings.mlr.press/v119/'],
'icml2025':['https://proceedings.mlr.press/v267/'],
'neurips2020':['https://papers.nips.cc/paper_files/paper/2020'],
'neurips2025':['https://papers.nips.cc/paper_files/paper/2025/vol38-main-conference']}
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 original={}
 for line in (ROOT/'results/paper_lists.tsv').read_text().splitlines():
  c,i,u=line.split('\t');original.setdefault(c,set()).add(i)
 features=[json.loads(x) for x in gzip.open(ROOT/'results/corpus_validation/features_numeric.jsonl.gz','rt')];summary={};allrows=[]
 for cell,urls in SOURCES.items():
  rows={};receipts=[]
  for n,url in enumerate(urls):
   p=CACHE/f'{cell}-{n}.html'
   if not p.exists():
    with urllib.request.urlopen(url,timeout=90) as response:p.write_bytes(response.read())
   b=p.read_bytes();h=b.decode('utf8');receipts.append({'url':url,'sha256':sha(b),'bytes':len(b),'private_snapshot':p.name})
   if cell.startswith('cvpr'):
    links=re.findall(r'href=["\']([^"\']+_paper\.pdf)["\']',h)
    for link in links:
     u=urllib.parse.urljoin(url,link);pid=pathlib.PurePosixPath(urllib.parse.urlparse(u).path).name.removesuffix('_paper.pdf');rows[pid]={'source_id':pid,'paper_url':'','pdf_url':u,'track':'main-conference'}
   elif cell.startswith('icml'):
    for vol,pid in re.findall(r'href=["\'](?:https?://proceedings\.mlr\.press)?/?(v\d+)/([^/"\']+)\.html["\']',h):
     if pid=='index':continue
     rows[pid]={'source_id':pid,'paper_url':f'https://proceedings.mlr.press/{vol}/{pid}.html','pdf_url':'','track':'volume-research-entry'}
   else:
    blocks=re.findall(r'<li\b[^>]*>[\s\S]*?</li>',h)
    for block in blocks:
     m=re.search(r'href=["\']([^"\']*/hash/([0-9a-f]{32})-Abstract[^"\']*\.html)["\']',block)
     if not m:continue
     track=re.search(r'data-track="([^"]+)"',block)
     rows[m[2]]={'source_id':m[2],'paper_url':urllib.parse.urljoin(url,m[1]),'pdf_url':'','track':track[1] if track else 'legacy-main-volume'}
  old=original.get(cell,set());catalog=set(rows);included=[x for x in features if x['corpus']==cell]
  pool=[r for pid,r in rows.items() if pid not in old and (not cell=='neurips2025' or r['track']=='conference')]
  summary[cell]={'catalog_entries':len(rows),'track_counts':{t:sum(r['track']==t for r in rows.values()) for t in sorted({r['track'] for r in rows.values()})},'original_manifest_n':len(old),'original_ids_found_in_current_catalog':len(old&catalog),'original_ids_not_found':len(old-catalog),'current_numeric_cohort_n':len(included),'new_source_id_disjoint_eligible_catalog_entries':len(pool),'new_all_track_entries':len(catalog-old),'snapshots':receipts}
  for pid,r in sorted(rows.items()):allrows.append({'cell':cell,**r,'in_original_manifest':int(pid in old),'expansion_catalog_eligible':int(pid not in old and (cell!='neurips2025' or r['track']=='conference'))})
  print(cell,summary[cell],flush=True)
 with (OUT/'catalog_entries.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 (OUT/'summary.json').write_text(json.dumps({'checked_utc_date':'2026-10-20','scope':'three venues; publication-year2020/2025 catalogs only; no PDF/outcome retrieval','original_manifest_sha256':sha((ROOT/'results/paper_lists.tsv').read_bytes()),'numeric_input_sha256':sha((ROOT/'results/corpus_validation/features_numeric.jsonl.gz').read_bytes()),'cells':summary},indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
