"""Validate two blind label files and report agreement only; no gold labels assumed."""
import argparse, collections, json, pathlib
LABELS={'interpretive','functional','other','uncertain'}
def load(path,packet):
 rows={}
 for line in path.read_text().splitlines():
  r=json.loads(line);i=r['item_id'];assert i in packet and i not in rows,'unknown/duplicate item'
  assert isinstance(r['uncertain'],bool) and isinstance(r['note'],str)
  for c in r['clauses']:
   assert c['label'] in LABELS and isinstance(c['uncertain'],bool)
   assert c['mainclause_attached'] is None or isinstance(c['mainclause_attached'],bool)
   assert 0<=c['start']<=c['verb_start']<c['verb_end']<=c['end']<=len(packet[i])
  rows[i]=r
 assert set(rows)==set(packet),'missing items'
 return rows
def signature(r):
 return tuple(sum(c['label']==label and c['mainclause_attached'] is True and not c['uncertain'] for c in r['clauses']) for label in ['interpretive','functional','other'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--private',type=pathlib.Path,required=True);p.add_argument('--reviewer-a',type=pathlib.Path,required=True);p.add_argument('--reviewer-b',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 packet={r['item_id']:r['sentence'] for r in map(json.loads,(a.private/'packet.jsonl').read_text().splitlines())};mapping={r['item_id']:r for r in map(json.loads,(a.private/'mapping.jsonl').read_text().splitlines())}
 x,y=load(a.reviewer_a,packet),load(a.reviewer_b,packet);out={'status':'two agent annotation files validated; agreement is not accuracy','strata':{},'performance':'No precision/recall: independent judgments are not adjudicated gold labels.'}
 for stratum in ['random','enriched']:
  ids=[i for i in packet if mapping[i]['stratum']==stratum]
  eligible=[i for i in ids if not x[i]['uncertain'] and not y[i]['uncertain'] and not any(c['uncertain'] or c['label']=='uncertain' or c['mainclause_attached'] is None for r in [x[i],y[i]] for c in r['clauses'])]
  out['strata'][stratum]={'n':len(ids),'jointly_certain_n':len(eligible),'uncertainty_excluded_n':len(ids)-len(eligible),'exact_semantic_count_agreement_n':sum(signature(x[i])==signature(y[i]) for i in eligible),'presence_agreement_n':sum(bool(sum(signature(x[i])))==bool(sum(signature(y[i]))) for i in eligible)}
 a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
