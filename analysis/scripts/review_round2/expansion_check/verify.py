"""Check complete PDF receipts, cached features and independent bootstrap arithmetic."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/review_round2/reproduction'))
import repro
RAW=Path.home()/'agent-papers-data/before-after-ai/expansion-20261020'
OUT=ROOT/'results/review_round2/expansion_run'
read=lambda n:json.loads((OUT/n).read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s,rec,rows,res=[read(n) for n in ['selection.json','receipts.json','paper_status.json','results.json']]
key=lambda r:(r['cell'],r['source_id'])
assert len(s['items'])==len(rec['items'])==len(rows)==1200
assert len({key(r) for r in rows})==1200
assert {key(r) for r in rows}=={key(r) for r in s['items']}=={key(r) for r in rec['items']}
assert rec['selection_sha256']==sha(OUT/'selection.json') and res['receipts_sha256']==sha(OUT/'receipts.json')
for r in rec['items']:
 p=RAW/'pdfs'/r['cell']/(r['source_id']+'.pdf');assert r['retrieved'] and p.stat().st_size==r['pdf_bytes'] and sha(p)==r['pdf_sha256']
assert len({r['pdf_sha256'] for r in rec['items']})==1200
old={r['pdf_sha256'] for r in csv.DictReader(open(ROOT/'results/review_round2/reproduction/pdf_sha256.csv')) if r['pdf_sha256']}
assert not old & {r['pdf_sha256'] for r in rec['items']}
for r in rows:
 f=repro.FEATURES.paper_features(json.loads((RAW/'text'/r['cell']/(r['source_id']+'.json')).read_text()))
 if f is None:assert r['status']=='excluded_under_1500_words'
 else:
  assert r['status']=='admitted'
  for k in ['words','ing_func_1k','ing_interp_core_1k','op_additionally_1k','op_notably_1k']:assert f[k]==r[k]
cells=sorted({r['cell'] for r in rows});ends=['ing_func_1k','ing_interp_core_1k','addnot_1k'];vs=['cvpr','icml','neurips']
rank={key(r):i for i,r in enumerate(s['items'])}
rows.sort(key=lambda r:rank[key(r)])
a={c:np.array([[r[k] for k in ends] for r in rows if r['cell']==c and r['status']=='admitted']) for c in cells}
rng=np.random.default_rng(20261020);ds=[]
for _ in range(2000):
 m={c:a[c][rng.integers(len(a[c]),size=len(a[c]))].mean(axis=0) for c in cells};ds.append([m[v+'2025']-m[v+'2020'] for v in vs])
ds=np.array(ds);primary=[]
for j,k in enumerate(ends):
 est=np.mean([a[v+'2025'][:,j].mean()-a[v+'2020'][:,j].mean() for v in vs]);ci=np.quantile(ds.mean(axis=1)[:,j],[.025,.975]);e=res['primary_equal_weight_mean_2025_minus_2020'][k]
 assert np.allclose([est,*ci],[e['est'],*e['ci95']],rtol=0,atol=1e-14)
 primary.append({'endpoint':k,'estimate':float(est),'ci95':ci.tolist()})
 for i,v in enumerate(vs):assert np.allclose(np.quantile(ds[:,i,j],[.025,.975]),res['venue_2025_minus_2020'][v][k]['ci95'],rtol=0,atol=1e-14)
out={'scope':'All PDF hashes and cached-text features checked; independent vectorized bootstrap. Not fresh extraction or an independent feature implementation.','pdfs_verified':1200,'exact_pdf_overlap_with_old_cache':0,'admitted':sum(r['status']=='admitted' for r in rows),'excluded':sum(r['status']=='excluded_under_1500_words' for r in rows),'primary':primary,'interval_match_atol':1e-14,'confirmed':all(x['ci95'][0]>0 for x in primary)}
(ROOT/'results/review_round2/expansion_check/statistics_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
