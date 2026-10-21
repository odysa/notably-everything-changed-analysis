#!/usr/bin/env python3
"""Offline audit of explicit first-page arXiv stamps; emits no paper identifiers."""
import argparse, collections, concurrent.futures, csv, datetime, hashlib, json, pathlib, re, subprocess

STAMP = re.compile(r'arXiv\s*:\s*([a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})v(\d+)\s*(?:\[[^\]\n]+\])?\s*(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})', re.I)

def parse(text, expected):
    matches=STAMP.findall(text)
    if not matches:
        return {'parse_status':'unparsed_stamp' if re.search(r'arXiv\s*:',text,re.I) else 'no_stamp'}
    parsed=set()
    for aid,version,date in matches:
        try: day=datetime.datetime.strptime(date,'%d %b %Y').date().isoformat()
        except ValueError: return {'parse_status':'invalid_date'}
        parsed.add((aid.lower(),int(version),day))
    if len(parsed)!=1: return {'parse_status':'ambiguous_multiple_stamps'}
    aid,version,day=parsed.pop()
    if aid != re.sub(r'v\d+$','',expected).lower(): return {'parse_status':'identifier_mismatch'}
    return {'parse_status':'parsed','arxiv_version':version,'stamp_date':day}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def tests():
    assert parse('arXiv:1511.06909v7 [cs.LG] 31 Mar 2016','1511.06909')['stamp_date']=='2016-03-31'
    assert parse('Published at ICLR 2024','1511.06909')['parse_status']=='no_stamp'
    assert parse('arXiv:1511.06909 31 Mar 2016','1511.06909')['parse_status']=='unparsed_stamp'
    assert parse('arXiv:1511.06909v7 [cs.LG] 31 Feb 2016','1511.06909')['parse_status']=='invalid_date'
    assert parse('arXiv:1511.06909v7 [cs.LG] 31 Mar 2016','1500.00000')['parse_status']=='identifier_mismatch'
    assert parse('arXiv:1511.06909v7 31 Mar 2016 arXiv:1511.06909v8 1 Apr 2016','1511.06909')['parse_status']=='ambiguous_multiple_stamps'

def main():
    p=argparse.ArgumentParser();p.add_argument('--workdir',type=pathlib.Path,required=True);p.add_argument('--list',type=pathlib.Path,required=True);p.add_argument('--features',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();tests()
    features=[json.loads(l) for l in a.features.open() if l.strip()]; active={(x['corpus'],str(x['id'])) for x in features}
    entries=[r for r in csv.reader(a.list.open(),delimiter='\t') if len(r)>=3 and re.search(r'https?://(?:www\.)?arxiv\.org/',r[2])]
    def audit(r):
        corpus,ident,url=r[:3]; path=a.workdir/'pdfs'/corpus/(ident+'.pdf')
        result={'corpus':corpus,'paper_hash':hashlib.sha256((corpus+ident).encode()).hexdigest()[:12],'active_feature_key':(corpus,ident) in active,'cache_present':path.is_file(),'pdf_sha256':'','parse_status':'missing_pdf','arxiv_version':'','stamp_date':'','after_2022_11_30':'','after_2023_03_14':''}
        if path.is_file():
            result['pdf_sha256']=sha(path)
            try:
                proc=subprocess.run(['pdftotext','-f','1','-l','1','-layout',str(path),'-'],capture_output=True,timeout=30)
                result.update(parse(proc.stdout.decode('utf-8',errors='replace'),ident) if proc.returncode==0 else {'parse_status':'extraction_error'})
            except (OSError,subprocess.TimeoutExpired):result['parse_status']='extraction_error'
            if result['parse_status']=='parsed':
                result['after_2022_11_30']=result['stamp_date']>'2022-11-30';result['after_2023_03_14']=result['stamp_date']>'2023-03-14'
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: rows=list(pool.map(audit,entries))
    aggregates=[]
    for cohort in sorted({r['corpus'] for r in rows}):
        for scope in ['listed','active']:
            rr=[r for r in rows if r['corpus']==cohort and (scope=='listed' or r['active_feature_key'])]
            aggregates.append({'corpus':cohort,'scope':scope,'n':len(rr),'cache_present':sum(r['cache_present'] for r in rr),'parsed':sum(r['parse_status']=='parsed' for r in rr),'unknown':sum(r['parse_status']!='parsed' for r in rr),'after_2022_11_30':sum(r['after_2022_11_30'] is True for r in rr),'after_2023_03_14':sum(r['after_2023_03_14'] is True for r in rr)})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for suffix,data in [('.csv',rows),('_cohorts.csv',aggregates)]:
        with pathlib.Path(str(a.out)+suffix).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    summary={'sources':{'list_sha256':sha(a.list),'features_sha256':sha(a.features)},'feature_rows':len(features),'unique_feature_keys':len(active),'listed_arxiv_entries':len(entries),'active_arxiv_entries':sum(r['active_feature_key'] for r in rows),'statuses_listed':dict(collections.Counter(r['parse_status'] for r in rows)),'statuses_active':dict(collections.Counter(r['parse_status'] for r in rows if r['active_feature_key'])),'cohorts':aggregates,'parser_checks':6,'workers':2,'pdftotext_version':subprocess.run(['pdftotext','-v'],capture_output=True,text=True).stderr.splitlines()[0]}
    pathlib.Path(str(a.out)+'.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
