#!/usr/bin/env python3
"""Contract checks for the frozen external validation artifacts."""
import hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[2];P=ROOT/'results/external_validation'
s=json.loads((P/'selection.json').read_text());r=json.loads((P/'retrieval.json').read_text());summary=json.loads((P/'summary.json').read_text());f=[json.loads(x) for x in (P/'features_numeric.jsonl').read_text().splitlines()]
assert len(r)==160 and len(f)==158 and len({x['paper_hash'] for x in f})==158
assert sum(x['year']<2023 for x in f)==79 and sum(x['year']>2023 for x in f)==79
assert len(s['selected'])==160 and len({x['paper_hash'] for x in s['selected']})==160
assert {x['paper_hash'] for x in r}=={x['paper_hash'] for x in s['selected']}
assert {x['paper_hash'] for x in f}=={x['paper_hash'] for x in r if x['status']=='included'}
assert all(set(x)==set(['paper_hash','year','words']+summary['style_features']) for x in f)
assert len(summary['style_features'])==33 and s['original_source_id_overlap']==0
for path,h in summary['hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
for y in [2019,2020,2024,2025]:
 assert sum(x['year']==y for x in r)==40
 assert sum(x['year']==y for x in f)==summary['counts'][str(y)]['included']
assert all(x['words']>=1500 for x in f)
import math
assert all(isinstance(x[k],(int,float)) and math.isfinite(x[k]) for x in f for k in summary['style_features'])
assert all(sum(x['year']==y for x in s['selected'])==40 for y in [2019,2020,2024,2025])
assert len(summary['results'])==5 and summary['bootstrap_replicates']==2000
print('External selection, eligibility, numeric schema, source hashes and result contracts passed.')

# An extraction incompatibility must refuse admission before model evaluation.
import importlib.util
spec=importlib.util.spec_from_file_location('external_emnlp',ROOT/'scripts/external_validation/emnlp.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.admit_statuses(r)
bad=[dict(x) for x in r];bad[0]['status']='error'
try:
 module.admit_statuses(bad)
except RuntimeError:
 pass
else:
 raise AssertionError('Extraction errors admitted to model evaluation')
try:
 module.admit_statuses(r[:-1])
except RuntimeError:
 pass
else:
 raise AssertionError('Incomplete retrieval admitted to model evaluation')
print('Error/incomplete-corpus refusal contracts passed.')
