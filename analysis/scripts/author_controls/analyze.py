#!/usr/bin/env python3
"""Exact Anthology paper IDs + explicit author IDs; never name-based linkage."""
import argparse, collections, datetime, gzip, hashlib, json, pathlib, urllib.request, xml.etree.ElementTree as ET
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
REV='0d4dd55c88a79724914e339ba3511086c05653d0'
SOURCES={'acl2019':'P19','acl2020':'2020.acl','acl2024':'2024.acl','acl2025':'2025.acl','acl2026':'2026.acl'}
METRICS=['sent_len','sent_cv','emdash_1k','style_lex_1k','famous_ai_words_1k','llm_1k','mattr','words']
def sha(b):return hashlib.sha256(b).hexdigest()
def h(s):return sha(s.encode())[:12]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=pathlib.Path,required=True);a=ap.parse_args();a.cache.mkdir(parents=True,exist_ok=True)
 out=ROOT/'results/author_controls';out.mkdir(exist_ok=True)
 features={x['paper_hash']:x for x in map(json.loads,gzip.open(ROOT/'results/corpus_validation/features_numeric.jsonl.gz','rt'))}
 sources={}; metadata={};coverage={}; mapping=[]
 for corpus,collection in SOURCES.items():
  url=f'https://raw.githubusercontent.com/acl-org/acl-anthology/{REV}/data/xml/{collection}.xml';p=a.cache/(collection+'.xml')
  if not p.exists():
   with urllib.request.urlopen(url,timeout=40) as r:p.write_bytes(r.read())
  data=p.read_bytes();sources[corpus]={'url':url,'sha256':sha(data),'bytes':len(data)}
  tree=ET.fromstring(data)
  for vol in tree.findall('volume'):
   for paper in vol.findall('paper'):
    pid=(collection+'-'+vol.attrib['id']+'.'+paper.attrib['id']) if '.' in collection else (collection+'-'+vol.attrib['id']+paper.attrib['id'].zfill(3))
    authors=paper.findall('author');ids=sorted(set(h('acl-author:'+x.attrib['id']) for x in authors if x.attrib.get('id')))
    assert (corpus,pid) not in metadata, 'duplicate official paper ID'
    metadata[(corpus,pid)]={'authors':ids,'author_count':len(authors),'explicit_count':sum('id' in x.attrib for x in authors)}
 selected=[]
 for line in open(ROOT/'results/paper_lists.tsv'):
  corpus,pid,url=line.strip().split('\t');key=h(corpus+pid)
  if corpus not in SOURCES or key not in features:continue
  row={'paper_hash':key,'corpus':corpus,'era':int(int(corpus[-4:])>=2024)}
  meta=metadata.get((corpus,pid));row.update(meta or {'authors':[],'author_count':0,'explicit_count':0});row['metadata_matched']=meta is not None
  selected.append(row)
 counts=collections.defaultdict(lambda:[set(),set()])
 for r in selected:
  for author in r['authors']:counts[author][r['era']].add(r['paper_hash'])
 recurring={au for au,ps in counts.items() if ps[0] and ps[1]}
 for r in selected:r['panel_authors']=sorted(recurring.intersection(r['authors']))
 panel=[r for r in selected if r['panel_authors']]
 for corpus in SOURCES:
  rows=[r for r in selected if r['corpus']==corpus]
  coverage[corpus]={'numeric_available':len(rows),'exact_metadata_matches':sum(r['metadata_matched'] for r in rows),'with_explicit_author_id':sum(bool(r['authors']) for r in rows),'panel_papers':sum(bool(r['panel_authors']) for r in rows),'all_author_occurrences':sum(r['author_count'] for r in rows),'explicit_id_occurrences':sum(r['explicit_count'] for r in rows)}
 results={}
 composition={}
 for corpus in SOURCES:
  composition[corpus]={}
  for metric in METRICS:
   allv=[features[r['paper_hash']][metric] for r in selected if r['corpus']==corpus]
   matchv=[features[r['paper_hash']][metric] for r in panel if r['corpus']==corpus]
   composition[corpus][metric]={'original_mean':float(np.mean(allv)),'panel_mean':float(np.mean(matchv)) if matchv else None}
 for metric in METRICS:
  vals={r['paper_hash']:features[r['paper_hash']].get(metric) for r in selected}
  def mean(rows,era):
   v=[vals[r['paper_hash']] for r in rows if r['era']==era and isinstance(vals[r['paper_hash']],(int,float)) and np.isfinite(vals[r['paper_hash']])];return float(np.mean(v)) if v else None
  # WLS on paper-author rows, each paper contributes total weight one.
  author_rows=collections.defaultdict(list)
  for r in panel:
   y=vals[r['paper_hash']]
   if not isinstance(y,(int,float)) or not np.isfinite(y):continue
   for au in r['panel_authors']:author_rows[au].append((r['era'],y,1/len(r['panel_authors'])))
  num=den=0.
  for rows in author_rows.values():
   t,y,w=np.array(rows).T;tm=np.average(t,weights=w);ym=np.average(y,weights=w);num+=np.sum(w*(t-tm)*(y-ym));den+=np.sum(w*(t-tm)**2)
  # Independent direct weighted least-squares check of the within transform.
  aus=sorted(author_rows);design=[];response=[];weights=[]
  for j,au in enumerate(aus):
   for t,y,w in author_rows[au]:
    design.append([t]+[float(i==j) for i in range(len(aus))]);response.append(y);weights.append(w)
  if den:
   sw=np.sqrt(weights);direct=np.linalg.lstsq(np.array(design)*sw[:,None],np.array(response)*sw,rcond=None)[0][0]
   assert np.isclose(direct,num/den), 'FE estimator disagreement'
  results[metric]={'all_pre':mean(selected,0),'all_post':mean(selected,1),'panel_pre':mean(panel,0),'panel_post':mean(panel,1),'author_FE_post_minus_pre':float(num/den) if den else None}
 # Coauthor connected components show why author rows cannot be independent.
 parent={r['paper_hash']:r['paper_hash'] for r in panel}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 for au,eras in counts.items():
  if au not in recurring:continue
  papers=list(eras[0]|eras[1]);p0=find(papers[0])
  for p in papers[1:]:parent[find(p)]=p0
 comps=collections.Counter(find(p) for p in parent)
 summary={'source_revision':REV,'retrieval_date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':sources,'numeric_input_sha256':sha((ROOT/'results/corpus_validation/features_numeric.jsonl.gz').read_bytes()),'coverage':coverage,'recurring_explicit_ids':len(recurring),'panel_unique_papers':len(panel),'paper_author_rows':sum(len(r['panel_authors']) for r in panel),'connected_component_sizes':sorted(comps.values(),reverse=True),'metrics':results,'composition_by_corpus':composition}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with (out/'mapping_hashed.jsonl').open('w') as f:
  for r in selected:f.write(json.dumps(r,sort_keys=True)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k not in ['sources','metrics','composition_by_corpus']},indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
