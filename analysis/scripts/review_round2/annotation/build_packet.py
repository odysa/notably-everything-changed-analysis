"""Frozen sampling; public manifest only, prose/mapping private."""
import argparse, gzip, hashlib, importlib.util, json, pathlib
ROOT=pathlib.Path(__file__).resolve().parents[3]
def sha(x): return hashlib.sha256(x.encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--private',type=pathlib.Path,required=True);a=p.parse_args()
 spec=importlib.util.spec_from_file_location('features',ROOT/'scripts/features.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
 rows=[json.loads(x) for x in gzip.open(ROOT/'results/corpus_validation/features_numeric.jsonl.gz','rt')]
 originals={(r['corpus'],sha(r['corpus']+r['id'])[:12]):r['id'] for r in map(json.loads,(a.cache/'features.jsonl').open())}
 packet=[];mapping=[];papers=[]
 for venue in ['acl','cvpr','osdi','vldb']:
  for year in [2020,2025]:
   corpus=f'{venue}{year}';pool=[r for r in rows if r['corpus']==corpus]
   if len(pool)<2: raise RuntimeError(f'No exact cohort: {corpus}')
   selected=sorted(pool,key=lambda r:(sha('20261020:'+r['id']),r['id']))[:2]
   for r in selected:
    original_id=originals[(corpus,r['id'])]
    path=a.cache/'text'/corpus/(original_id+'.json')
    if not path.exists():raise RuntimeError(f'Missing fixed text {path}; no replacement')
    rec=json.loads(path.read_text());ss=[s for para in f.drop_policy(rec) for s in f.sentences(para)]
    if len(ss)<20:raise RuntimeError('Fixed paper has fewer than 20 sentences')
    key=f"{corpus}:{r['id']}"
    rand=sorted(range(len(ss)),key=lambda i:sha(f'20261020:random:{key}:{i}'))[:20]
    enriched=sorted([i for i,s in enumerate(ss) if i not in rand and sum(f.ing_riders([s])[:4])],key=lambda i:sha(f'20261020:enrich:{key}:{i}'))[:6]
    ph=sha(key)
    papers.append({'paper_hash':ph,'random_count':len(rand),'enriched_count':len(enriched),'text_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for stratum,indices in [('random',rand),('enriched',enriched)]:
     for i in indices:
      # Opaque IDs do not encode source or sampling stratum.
      item=sha(f'20261020:opaque:{key}:{i}')[:24]
      packet.append({'item_id':item,'sentence':ss[i]})
      mapping.append({'item_id':item,'corpus':corpus,'id':original_id,'sentence_index':i,'stratum':stratum,'detector_counts':list(f.ing_riders([ss[i]])[:4])})
 a.private.mkdir(parents=True,exist_ok=True)
 for name,data in [('packet.jsonl',packet),('mapping.jsonl',mapping)]:
  (a.private/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(data,key=lambda r:r['item_id'])))
 manifest={'protocol_sha256':hashlib.sha256((ROOT/'docs/review-round2/annotation/PROTOCOL.md').read_bytes()).hexdigest(),'features_sha256':hashlib.sha256((ROOT/'scripts/features.py').read_bytes()).hexdigest(),'papers':papers,'paper_count':len(papers),'random_count':sum(p['random_count'] for p in papers),'enriched_count':sum(p['enriched_count'] for p in papers),'packet_sha256':hashlib.sha256((a.private/'packet.jsonl').read_bytes()).hexdigest(),'item_ids':[r['item_id'] for r in sorted(packet,key=lambda r:r['item_id'])],'annotation_status':'not yet annotated; no performance estimates'}
 (ROOT/'results/review_round2/annotation/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:manifest[k] for k in ['paper_count','random_count','enriched_count','packet_sha256']}))
if __name__=='__main__':main()
