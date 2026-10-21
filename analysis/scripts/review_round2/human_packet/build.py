"""Create a private human review handoff; no labels or accuracy estimates."""
import argparse,ast,gzip,hashlib,importlib.util,json,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[3]
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--private',type=pathlib.Path,required=True);a=p.parse_args()
 spec=importlib.util.spec_from_file_location('features',ROOT/'scripts/features.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
 tree=ast.parse((ROOT/'scripts/analyze.py').read_text());venues=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VENUES' for t in n.targets))
 cohorts={c:era for v,(_,pre,post) in venues.items() if not v.endswith('*') for era,cs in [('pre',pre),('post',post)] for c in cs}
 originals={(r['corpus'],sha(r['corpus']+r['id'])[:12]):r['id'] for r in map(json.loads,(a.cache/'features.jsonl').open())}
 rows=[r for r in map(json.loads,gzip.open(ROOT/'results/corpus_validation/features_numeric.jsonl.gz','rt')) if r['corpus'] in cohorts]
 sentences={};targets={'pre':[],'post':[]}
 for r in rows:
  corpus=r['corpus'];hid=r['id'];path=a.cache/'text'/corpus/(originals[(corpus,hid)]+'.json')
  if not path.exists():raise RuntimeError(f'Fixed source missing: {path}; no replacement')
  rec=json.loads(path.read_text());ss=[s for para in f.drop_policy(rec) for s in f.sentences(para)];sentences[(corpus,hid)]=ss
  for si,s in enumerate(ss):
   for m in re.finditer(r',\s+(?:thereby |thus |further |ultimately |effectively )?([a-z]+ing)\b',s):
    if m.start()>=25 and m[1] in f.INTERP_CORE:
     targets[cohorts[corpus]].append(dict(corpus=corpus,paper_hash=hid,sentence_index=si,verb_start=m.start(1),verb_end=m.end(1),rank=sha(f'20261020:core:{corpus}:{hid}:{si}:{m.start(1)}:{m.end(1)}')))
 selected=[];used=set();poolcounts={}
 for era in ['pre','post']:
  poolcounts[era]=len({(r['corpus'],r['paper_hash']) for r in targets[era]})
  if poolcounts[era]<40:raise RuntimeError(f'{era}: only {poolcounts[era]} distinct eligible core papers; stopped without fallback')
  taken=0
  for r in sorted(targets[era],key=lambda r:r['rank']):
   key=(r['corpus'],r['paper_hash'])
   if key in used:continue
   used.add(key);selected.append(dict(r,era=era,stratum='core'));taken+=1
   if taken==40:break
 for era in ['pre','post']:
  candidates=[]
  for (corpus,hid),ss in sentences.items():
   if cohorts[corpus]!=era or (corpus,hid) in used:continue
   for si in range(len(ss)):candidates.append(dict(corpus=corpus,paper_hash=hid,sentence_index=si,rank=sha(f'20261020:control:{corpus}:{hid}:{si}')))
  taken=0
  for r in sorted(candidates,key=lambda r:r['rank']):
   key=(r['corpus'],r['paper_hash'])
   if key in used:continue
   used.add(key);selected.append(dict(r,era=era,stratum='control'));taken+=1
   if taken==20:break
  if taken<20:raise RuntimeError('Insufficient disjoint control papers')
 packet=[]
 for r in selected:
  i=sha('20261020:human-opaque:'+r['rank'])[:24];r['item_id']=i
  packet.append(dict(item_id=i,sentence=sentences[(r['corpus'],r['paper_hash'])][r['sentence_index']]))
 a.private.mkdir(parents=True,exist_ok=True)
 for name,data in [('packet.jsonl',packet),('mapping.jsonl',selected)]:
  (a.private/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(data,key=lambda r:r['item_id'])))
 form=[dict(item_id=r['item_id'],clauses=[],uncertain=None,note='') for r in sorted(packet,key=lambda r:r['item_id'])]
 (a.private/'reviewer_form.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in form))
 manifest=dict(status='Prepared for real human review; no labels or accuracy estimates',counts={'core_per_era':40,'control_per_era':20,'total':len(packet),'distinct_papers':len(used)},eligible_core_papers_by_era=poolcounts,protocol_sha256=digest(ROOT/'docs/review-round2/human_packet/PROTOCOL.md'),features_sha256=digest(ROOT/'scripts/features.py'),mapping_source_sha256=digest(ROOT/'scripts/analyze.py'),numeric_map_sha256=digest(ROOT/'results/corpus_validation/features_numeric.jsonl.gz'),private_file_sha256={n:digest(a.private/n) for n in ['packet.jsonl','mapping.jsonl','reviewer_form.jsonl']},item_ids=[r['item_id'] for r in sorted(packet,key=lambda r:r['item_id'])])
 (ROOT/'results/review_round2/human_packet/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:manifest[k] for k in ['counts','eligible_core_papers_by_era']}))
if __name__=='__main__':main()
