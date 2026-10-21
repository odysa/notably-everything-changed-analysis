#!/usr/bin/env python3
"""Frozen, bounded publisher-format external confirmation; no prose in outputs."""
import argparse, hashlib, importlib.util, json, os, pathlib, sys, time, urllib.parse, urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[3]
OUT=ROOT/'results/review_round2/external'
DATA_DIR=pathlib.Path(os.environ.get('BEFORE_AFTER_AI_DATA_DIR', pathlib.Path.home()/'.cache'/'before-after-ai'))
CACHE=DATA_DIR/'review_round2_external'
YEARS=(2019,2020,2024,2025); SEED=20261020
KEYS=('ing_func_1k','ing_interp_core_1k','op_additionally_1k','op_notably_1k')
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def get(url):
 for n in range(3):
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ResearchValidation/1.0 (bounded academic corpus check)'}),timeout=60) as r:return r.read()
  except Exception:
   if n==2:raise
   time.sleep(3*(n+1))
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def select():
 selected=[];sources={}
 for year in YEARS:
  q=f'journal:"PLOS Genetics" AND article_type:"research article" AND publication_date:[{year}-01-01T00:00:00Z TO {year}-12-31T23:59:59Z]'
  docs=[];start=0
  while True:
   url='https://api.plos.org/search?'+urllib.parse.urlencode({'q':q,'fl':'id,journal,article_type,publication_date','wt':'json','rows':1000,'start':start,'sort':'id asc'})
   mp=CACHE/'metadata'/f'{year}-{start}.json';b=mp.read_bytes() if mp.exists() else get(url);dump(mp,json.loads(b));d=json.loads(b)['response'];docs+=d['docs'];start+=len(d['docs'])
   if start>=d['numFound']:break
   if not d['docs']:raise RuntimeError('incomplete metadata paging')
   time.sleep(1)
  valid={}
  for d in docs:
   types=d.get('article_type',[]);types=[types] if isinstance(types,str) else types
   assert d['journal']=='PLOS Genetics' and 'research article' in [t.lower() for t in types] and d['publication_date'].startswith(str(year))
   doi=d['id']
   if '/' in doi.split('/',1)[1]:continue # Solr partial-document IDs are not article DOIs
   assert doi.startswith('10.1371/journal.pgen.')
   valid[doi]=d
  assert len(valid)>=20
  chosen=sorted(valid,key=lambda doi:sha(f'{SEED}:{doi}'.encode()))[:20]
  selected += [{'doi':doi,'year':year,'paper_hash':sha(('round2-plos:'+doi).encode())[:16]} for doi in chosen]
  sources[str(year)]={'query':q,'eligible':len(valid),'returned':len(docs),'metadata_hashes':{p.name:sha(p.read_bytes()) for p in sorted((CACHE/'metadata').glob(f'{year}-*.json'))}}
  print(year,len(valid),flush=True);time.sleep(1)
 dump(CACHE/'selected_private.json',selected)
 dump(OUT/'selection.json',{'sources':sources,'selected':[ {k:v for k,v in r.items() if k!='doi'} for r in selected],'private_manifest_sha256':sha((CACHE/'selected_private.json').read_bytes()),'n':len(selected)})

def run():
 import numpy as np
 helper=module('_common',ROOT/'scripts/corpus_validation/extractor_snapshot.py');sys.modules['_common']=helper
 ex=module('original_extract',ROOT/'scripts/extract.py');feat=module('original_features',ROOT/'scripts/features.py')
 selected=json.loads((CACHE/'selected_private.json').read_text());assert len(selected)==80
 assert sha((CACHE/'selected_private.json').read_bytes())==json.loads((OUT/'selection.json').read_text())['private_manifest_sha256']
 rows=[];statuses=[]
 for i,r in enumerate(selected):
  pid=r['paper_hash'];pdf=CACHE/'pdfs'/f'{pid}.pdf';dst=CACHE/'text'/f'{pid}.json';pdf.parent.mkdir(parents=True,exist_ok=True);dst.parent.mkdir(parents=True,exist_ok=True)
  s={'paper_hash':pid,'year':r['year']}
  try:
   if not pdf.exists():
    b=get('https://journals.plos.org/plosgenetics/article/file?'+urllib.parse.urlencode({'id':r['doi'],'type':'printable'}));assert b.startswith(b'%PDF-'),'nonPDF';pdf.write_bytes(b);time.sleep(1)
   s['pdf_sha256']=sha(pdf.read_bytes())
   if not dst.exists():assert ex.one((f'plosgenetics{r["year"]}',str(pdf),str(dst))) is not None,'extraction failed'
   f=feat.paper_features(json.loads(dst.read_text()))
   if f is None:s['status']='excluded_under_1500_words'
   else:
    assert all(k in f and np.isfinite(f[k]) for k in KEYS)
    rows.append({'paper_hash':pid,'year':r['year'],'words':f['words'],**{k:f[k] for k in KEYS}});s['status']='included'
  except Exception as e:s['status']='error';s['error_type']=type(e).__name__
  statuses.append(s);dump(OUT/'retrieval.json',statuses)
  print(i+1,s['status'],flush=True)
 if len(statuses)!=80 or any(s['status']=='error' for s in statuses):
  dump(OUT/'summary.json',{'status':'stopped_retrieval_or_extraction_failure','n_processed':len(statuses),'n_errors':sum(s['status']=='error' for s in statuses),'confirmatory_estimates_computed':False});return
 with (OUT/'features_numeric.jsonl').open('w') as o:
  for r in rows:o.write(json.dumps(r,sort_keys=True)+'\n')
 for r in rows:r['addnot_1k']=r['op_additionally_1k']+r['op_notably_1k']
 metrics=('ing_func_1k','ing_interp_core_1k','addnot_1k');groups={y:[r for r in rows if r['year']==y] for y in YEARS};assert all(groups.values())
 def estimate(rs,k):return float(np.mean([r[k] for r in rs if r['year']>2023])-np.mean([r[k] for r in rs if r['year']<2023]))
 rng=np.random.default_rng(SEED);boots={k:[] for k in metrics}
 for _ in range(2000):
  rs=[g[j] for g in groups.values() for j in rng.integers(len(g),size=len(g))]
  for k in metrics:boots[k].append(estimate(rs,k))
 results={k:{'post_minus_pre_1k':estimate(rows,k),'conditional_year_stratified_paper_bootstrap_95ci':np.quantile(boots[k],[.025,.975]).tolist()} for k in metrics}
 dump(OUT/'summary.json',{'status':'completed','confirmatory_estimates_computed':True,'confirmation':all(v['conditional_year_stratified_paper_bootstrap_95ci'][0]>0 for v in results.values()),'n_included':len(rows),'counts':{str(y):{'selected':20,'included':len(groups[y]),'excluded_under_1500_words':sum(s['year']==y and s['status']=='excluded_under_1500_words' for s in statuses)} for y in YEARS},'year_means':{str(y):{k:float(np.mean([r[k] for r in groups[y]])) for k in metrics} for y in YEARS},'results':results,'bootstrap':{'replicates':2000,'seed':SEED},'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [ROOT/'scripts/extract.py',ROOT/'scripts/features.py',ROOT/'scripts/corpus_validation/extractor_snapshot.py',ROOT/'docs/review-round2/external/PROTOCOL.md',OUT/'selection.json']}})
 print(json.dumps(results,indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('phase',choices=['select','run']);args=a.parse_args();CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);globals()[args.phase]()
