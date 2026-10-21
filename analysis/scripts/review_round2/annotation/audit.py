"""Descriptive blinded-agent disagreement audit, never gold accuracy."""
import argparse,collections,hashlib,importlib.util,json,pathlib,re
from evaluate import load
ROOT=pathlib.Path(__file__).resolve().parents[3]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def detector(s,f):
 out=[]
 for m in re.finditer(r',\s+(?:thereby |thus |further |ultimately |effectively )?([a-z]+ing)\b',s):
  if m.start()<25:continue
  v=m[1]
  if v in f.INTERP:label='interpretive'
  elif v in f.FUNC:label='functional'
  elif v not in ('including','using','following','according','during','being','having','regarding','string','nothing','something','anything','everything','padding','bring','thing','king','ring','spring'):label='other'
  else:continue
  out.append(dict(verb_start=m.start(1),verb_end=m.end(1),label=label,core=v in f.INTERP_CORE,lexical_group=('FUNC' if v in f.FUNC else 'INTERP_CORE' if v in f.INTERP_CORE else 'INTERP_WEAK' if v in f.INTERP_WEAK else 'residual_other')))
 return out
def key(c):return(c['verb_start'],c['verb_end'])
def confident(r,c):return not r['uncertain'] and not c['uncertain'] and c['label']!='uncertain' and c['mainclause_attached'] is not None

def main():
 p=argparse.ArgumentParser();p.add_argument('--private',type=pathlib.Path,required=True);a=p.parse_args();d=a.private
 packet={r['item_id']:r['sentence'] for r in map(json.loads,(d/'packet.jsonl').read_text().splitlines())};mapping={r['item_id']:r for r in map(json.loads,(d/'mapping.jsonl').read_text().splitlines())}
 spec=importlib.util.spec_from_file_location('features',ROOT/'scripts/features.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
 labels={name:load(d/f'annotations_{name}.jsonl',packet) for name in ['A','B']}
 groups=collections.defaultdict(collections.Counter);queue=[]
 for i,s in packet.items():
  meta=mapping[i];corpus=meta['corpus'];venue=re.sub(r'\d','',corpus);era='pre2020' if corpus.endswith('2020') else 'post2025';stratum=meta['stratum'];ds=detector(s,f);dm={key(c):c for c in ds};xs={n:labels[n][i] for n in labels};cms={n:{key(c):c for c in r['clauses'] if confident(r,c)} for n,r in xs.items()};joint={k:c for k,c in cms['A'].items() if k in cms['B'] and (c['label'],c['mainclause_attached'])==(cms['B'][k]['label'],cms['B'][k]['mainclause_attached'])}
  counts=collections.Counter(items=1,detector_heads=len(ds),joint_confident_heads=len(joint))
  for n,r in xs.items():
   counts[f'{n}_sentence_uncertain']+=r['uncertain'];counts[f'{n}_clauses']+=len(r['clauses']);counts[f'{n}_uncertain_or_unresolved_clauses']+=sum(c['uncertain'] or c['label']=='uncertain' or c['mainclause_attached'] is None for c in r['clauses']);counts[f'{n}_confident_clauses']+=len(cms[n]);counts[f'{n}_all_uncertainty_excluded_clauses']+=len(r['clauses'])-len(cms[n])
   for c in r['clauses']:
    if not s[c['verb_start']:c['verb_end']].lower().endswith('ing'):counts[f'{n}_non_ing_annotated_heads']+=1
   for k,c in cms[n].items():
    if s[k[0]:k[1]].lower().endswith('ing'):counts[f'{n}_confident_ing_only_sensitivity_clauses']+=1
    counts[f'{n}_confident_{c["label"]}_attached_{c["mainclause_attached"]}']+=1
    if c['label']=='interpretive' and c['mainclause_attached'] and s[k[0]:k[1]].lower() in f.INTERP_CORE:counts[f'{n}_semantic_interpretive_core_mask']+=1
   for k,c in dm.items():
    if k not in cms[n]:counts[f'{n}_detector_without_confident_exact_head']+=1
    elif cms[n][k]['label']!=c['label'] or not cms[n][k]['mainclause_attached']:counts[f'{n}_aligned_detector_semantic_or_attachment_disagreement']+=1
    else:counts[f'{n}_aligned_detector_category_attachment_agreement']+=1
  for k,c in dm.items():
   prefix='lexical_'+c['lexical_group']
   counts[prefix+'_detector_heads']+=1
   for n in ['A','B']:
    rc=cms[n].get(k)
    status='without_confident_head' if rc is None else 'attachment_disagreement' if not rc['mainclause_attached'] else 'semantic_agreement' if rc['label']==c['label'] else 'semantic_difference'
    counts[prefix+'_'+n+'_'+status]+=1
   rc=joint.get(k)
   status='without_joint_confident_head' if rc is None else 'attachment_disagreement' if not rc['mainclause_attached'] else 'semantic_agreement' if rc['label']==c['label'] else 'semantic_difference'
   counts[prefix+'_joint_'+status]+=1
   if k not in joint:counts['detector_without_joint_confident_head']+=1
   elif joint[k]['label']!=c['label'] or not joint[k]['mainclause_attached']:counts['detector_joint_confident_disagreement']+=1
   else:counts['detector_joint_confident_agreement']+=1
  for group in [stratum,f'{stratum}/{era}',f'{stratum}/{venue}/{era}']:groups[group].update(counts)
  # Queue any reviewer uncertainty, nonidentical annotations, or detector mismatch.
  normalized=lambda r:sorted((key(c),c['label'],c['mainclause_attached'],c['uncertain']) for c in r['clauses'])
  if any(r['uncertain'] or any(c['uncertain'] or c['label']=='uncertain' or c['mainclause_attached'] is None for c in r['clauses']) for r in xs.values()) or normalized(xs['A'])!=normalized(xs['B']) or counts['detector_without_joint_confident_head'] or counts['detector_joint_confident_disagreement']:
   queue.append({'item_id':i,'sentence':s,'annotations':xs,'detector':ds})
 qp=d/'human_adjudication_queue.jsonl';qp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in queue))
 out={'description':'Descriptive same-model agent pilot; agreement is not accuracy; no human gold or model diversity guarantee. Sentence uncertainty is not proven extraction damage. Unmatched heads are not proven false positives. No precision or recall.','input_sha256':{name:digest(d/name) for name in ['packet.jsonl','annotations_A.jsonl','annotations_B.jsonl']},'audit_protocol_sha256':digest(ROOT/'docs/review-round2/annotation/AUDIT_PROTOCOL.md'),'groups':{k:{**dict(sorted(v.items())),**{f'{n}_sentence_uncertainty_rate':v[f'{n}_sentence_uncertain']/v['items'] for n in ['A','B']}} for k,v in sorted(groups.items())},'private_adjudication_queue':{'count':len(queue),'sha256':digest(qp)}}
 (ROOT/'results/review_round2/annotation/descriptive_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'queue':len(queue),'random':out['groups']['random'],'enriched':out['groups']['enriched']}))
if __name__=='__main__':main()
