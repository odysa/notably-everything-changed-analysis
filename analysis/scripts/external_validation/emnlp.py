#!/usr/bin/env python3
"""Prospectively frozen EMNLP check; raw material stays in private cache."""
import argparse, ast, csv, gzip, hashlib, importlib.util, json, pathlib, sys, time, urllib.request, xml.etree.ElementTree as ET
ROOT=pathlib.Path(__file__).resolve().parents[2]; OUT=ROOT/'results/external_validation'
REV='0d4dd55c88a79724914e339ba3511086c05653d0'; SEED=20261019
SOURCES={2019:('D19','1'),2020:('2020.emnlp','main'),2024:('2024.emnlp','main'),2025:('2025.emnlp','main')}
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def fetch(cache):
 (cache/"metadata").mkdir(exist_ok=True)
 for coll,_ in SOURCES.values():
  p=cache/"metadata"/f"{coll}.xml"
  if not p.exists():
   with urllib.request.urlopen(f"https://raw.githubusercontent.com/acl-org/acl-anthology/{REV}/data/xml/{coll}.xml",timeout=40) as r:p.write_bytes(r.read())

def select(cache):
 original={r.split('\t')[1] for r in (ROOT/'results/paper_lists.tsv').read_text().splitlines()}; chosen=[];sources={}
 for year,(coll,vid) in SOURCES.items():
  path=cache/'metadata'/f'{coll}.xml';b=path.read_bytes();tree=ET.fromstring(b);vol=next(v for v in tree.findall('volume') if v.attrib['id']==vid)
  assert vol.findtext('meta/year')==str(year) and 'emnlp' in [v.text for v in vol.findall('meta/venue')]
  candidates=[]
  for p in vol.findall('paper'):
   if not p.findall('author') or not p.findtext('pages') or int(p.attrib['id'])==0:continue
   pid=f'{coll}-{vid}{p.attrib["id"].zfill(3)}' if year==2019 else f'{coll}-{vid}.{p.attrib["id"]}'
   assert pid not in original,'source overlap'
   candidates.append({'id':pid,'year':year,'paper_hash':sha(('external-emnlp:'+pid).encode())[:12]})
  chosen+=sorted(candidates,key=lambda r:sha((str(SEED)+':'+r['id']).encode()))[:40]
  sources[str(year)]={'collection':coll,'volume':vid,'metadata_sha256':sha(b),'eligible_entries':len(candidates),'long_short_labels':False,'url':f'https://raw.githubusercontent.com/acl-org/acl-anthology/{REV}/data/xml/{coll}.xml'}
 assert len(chosen)==160 and len({r['paper_hash'] for r in chosen})==160
 dump(cache/'selected_private.json',chosen)
 dump(OUT/'selection.json',{'revision':REV,'sources':sources,'selected':[{k:v for k,v in r.items() if k!='id'} for r in chosen],'original_source_id_overlap':0,'private_selection_sha256':sha((cache/'selected_private.json').read_bytes()),'length_note':'No explicit long/short labels in selected XML volumes; main research entries include mixed lengths. 2019 is joint EMNLP-IJCNLP; other years EMNLP main. No page-count inference.'})

def admit_statuses(statuses):
 """A partial/error corpus cannot enter the frozen model evaluation."""
 if len(statuses)!=160 or any(r.get("status") not in {"included","excluded_under_1500_words"} for r in statuses):
  raise RuntimeError("External evaluation stopped: incomplete or incompatible retrieval/extraction; inspect retrieval.json")

def run(cache):
 import numpy as np
 import pandas as pd
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import roc_auc_score
 val=module('validation',ROOT/'scripts/stat_validation/validate.py');_,_,train,defs=val.load(ROOT,ROOT/'scripts',ROOT/'results/corpus_validation/features_numeric.jsonl.gz');train=train[train.era.isin(['pre','post'])];fs=defs['STYLE'];assert len(fs)==33
 # Inject exact committed helper into unchanged original extractor.
 helper=module('_common',ROOT/'scripts/corpus_validation/extractor_snapshot.py');sys.modules['_common']=helper
 ex=module('original_extract',ROOT/'scripts/extract.py');feat=module('original_features',ROOT/'scripts/features.py')
 rows=[];statuses=[]
 for i,r in enumerate(json.loads((cache/'selected_private.json').read_text())):
  y=r['year'];pid=r['id'];pdf=cache/'pdfs'/f'{pid}.pdf';dst=cache/'text'/f'{pid}.json';pdf.parent.mkdir(exist_ok=True);dst.parent.mkdir(exist_ok=True)
  status={'paper_hash':r['paper_hash'],'year':y,'status':None}
  try:
   if not pdf.exists():
    with urllib.request.urlopen(f'https://aclanthology.org/{pid}.pdf',timeout=60) as response:b=response.read()
    assert b.startswith(b'%PDF-'),'not PDF';pdf.write_bytes(b);time.sleep(.5)
   status['pdf_sha256']=sha(pdf.read_bytes())
   if not dst.exists():assert ex.one((f'emnlp{y}',str(pdf),str(dst))) is not None,'extraction failed'
   f=feat.paper_features(json.loads(dst.read_text()))
   if f is None:status['status']='excluded_under_1500_words'
   else:
    assert all(k in f and np.isfinite(f[k]) for k in fs),'incompatible STYLE features'
    rows.append({'paper_hash':r['paper_hash'],'year':y,'words':f['words'],**{k:f[k] for k in fs}});status['status']='included'
  except Exception as e:status['status']='error';status['error_type']=type(e).__name__
  statuses.append(status);dump(OUT/'retrieval.json',statuses)
  if (i+1)%20==0:print('processed',i+1,flush=True)
 admit_statuses(statuses)
 with (OUT/'features_numeric.jsonl').open('w') as out:
  for r in rows:out.write(json.dumps(r,sort_keys=True)+'\n')
 test=pd.DataFrame(rows);test['venue']='EMNLP';test['era']=np.where(test.year<2023,'pre','post')
 clf=LogisticRegression(C=.3,max_iter=2000).fit(val.transform(train,train,fs,'pooled'),(train.era=='post').astype(int))
 test['score']=clf.predict_proba(val.transform(train,test,fs,'pooled'))[:,1];test['addnot_1k']=test.op_additionally_1k+test.op_notably_1k
 metrics=['ing_func_1k','ing_interp_core_1k','addnot_1k'];rng=np.random.default_rng(SEED)
 def estimates(d):
  pre=d[d.year<2023];post=d[d.year>2023]
  return {'auc':float(roc_auc_score(d.year>2023,d.score)),**{m:float(post[m].mean()-pre[m].mean()) for m in metrics}}
 groups={'pooled':test}
 for a in [2019,2020]:
  for b in [2024,2025]:groups[f'{a}_vs_{b}']=test[test.year.isin([a,b])]
 result={}
 for label,d in groups.items():
  point=estimates(d);boot=[estimates(pd.concat([g.iloc[rng.integers(len(g),size=len(g))] for _,g in d.groupby('year')])) for _ in range(2000)]
  result[label]={k:{'estimate':v,'conditional_paper_bootstrap_95ci':np.quantile([x[k] for x in boot],[.025,.975]).tolist()} for k,v in point.items()}
 test[['paper_hash','year','score']].to_csv(OUT/'scores.csv',index=False)
 dump(OUT/'summary.json',{'training_n':len(train),'training_pre_n':int((train.era=='pre').sum()),'style_features':fs,'counts':{str(y):{'selected':40,'included':int((test.year==y).sum()),'statuses':{s:sum(r['year']==y and r['status']==s for r in statuses) for s in sorted({r['status'] for r in statuses})}} for y in SOURCES},'results':result,'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [ROOT/'scripts/features.py',ROOT/'scripts/extract.py',ROOT/'scripts/stats.py',ROOT/'scripts/corpus_validation/extractor_snapshot.py',ROOT/'results/corpus_validation/features_numeric.jsonl.gz',OUT/'PROTOCOL.md',OUT/'selection.json']},'bootstrap_replicates':2000,'seed':SEED})
 print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['fetch','select','run']);p.add_argument('--cache',type=pathlib.Path,required=True);a=p.parse_args();a.cache.mkdir(parents=True,exist_ok=True);globals()[a.phase](a.cache)
