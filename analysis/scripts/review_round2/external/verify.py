#!/usr/bin/env python3
"""Independent numerical and public-payload contract audit (no raw data needed)."""
import json,pathlib
import numpy as np
P=pathlib.Path(__file__).resolve().parents[3]/'results/review_round2/external'
s=json.loads((P/'summary.json').read_text());r=json.loads((P/'retrieval.json').read_text())
assert len(r)==80 and all(x['status'] in {'included','excluded_under_1500_words'} for x in r)
a=[json.loads(x) for x in (P/'features_numeric.jsonl').read_text().splitlines()]
allowed={'paper_hash','year','words','ing_func_1k','ing_interp_core_1k','op_additionally_1k','op_notably_1k'}
assert len(a)==s['n_included']==sum(x['status']=='included' for x in r)
assert all(set(x)==allowed and x['words']>=1500 for x in a)
metrics=('ing_func_1k','ing_interp_core_1k','addnot_1k')
for x in a:x['addnot_1k']=x['op_additionally_1k']+x['op_notably_1k']
years=(2019,2020,2024,2025);v={y:np.array([[x[k] for k in metrics] for x in a if x['year']==y]) for y in years}
def diff(z):return np.concatenate([z[2024],z[2025]]).mean(0)-np.concatenate([z[2019],z[2020]]).mean(0)
point=diff(v);rng=np.random.default_rng(20261020)
b=np.array([diff({y:z[rng.integers(len(z),size=len(z))] for y,z in v.items()}) for _ in range(2000)])
for j,k in enumerate(metrics):
 assert np.isclose(point[j],s['results'][k]['post_minus_pre_1k'],rtol=0,atol=1e-12)
 assert np.allclose(np.quantile(b[:,j],[.025,.975]),s['results'][k]['conditional_year_stratified_paper_bootstrap_95ci'],rtol=0,atol=1e-12)
assert s['confirmation']==bool(np.all(np.quantile(b,.025,axis=0)>0))
print('80 statuses, numeric-only rows, three point estimates and all 2000-bootstrap intervals verified')
