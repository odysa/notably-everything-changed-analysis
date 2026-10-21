#!/usr/bin/env python3
"""Bounded, offline-safe parsing of official pinned-version arXiv sources."""
import argparse, csv, gzip, hashlib, importlib.util, io, json, pathlib, re, statistics, tarfile, time, urllib.request
METRICS=['ing_interp_core_1k','ing_func_1k','boost_1k','op_additionally_1k','op_notably_1k','emdash_1k','this_closer_1k','style_lex_1k','famous_ai_words_1k']
def sha(b):return hashlib.sha256(b).hexdigest()
def safe_tex(blob):
    files={}
    try:
        with tarfile.open(fileobj=io.BytesIO(blob),mode='r:*') as t:
            for m in t.getmembers():
                p=pathlib.PurePosixPath(m.name)
                if p.is_absolute() or '..' in p.parts or m.issym() or m.islnk():raise ValueError('unsafe_archive')
                if m.isfile() and p.suffix.lower()=='.tex':
                    if m.size>5_000_000:raise ValueError('oversize_tex')
                    files[str(p)]=t.extractfile(m).read().decode('utf-8',errors='replace')
    except tarfile.ReadError:
        text=gzip.decompress(blob) if blob[:2]==b'\x1f\x8b' else blob
        if b'\\documentclass' not in text:raise ValueError('not_tex_archive')
        files['main.tex']=text.decode('utf-8',errors='replace')
    if not files:raise ValueError('no_tex')
    roots=sorted(k for k,v in files.items() if '\\begin{document}' in v)
    if len(roots)!=1:raise ValueError('ambiguous_root')
    visited=set()
    def expand(k):
        if k in visited:return ''
        visited.add(k);s=re.sub(r'(?<!\\)%[^\n]*','',files[k])
        def sub(m):
            target=str(pathlib.PurePosixPath(k).parent/(m[1] if m[1].endswith('.tex') else m[1]+'.tex'))
            return expand(target) if target in files else ''
        return re.sub(r'\\(?:input|include)\s*\{([^}]+)\}',sub,s)
    return expand(roots[0]),len(files),len(visited)
def prose(s):
    s=s.split('\\begin{document}',1)[-1].split('\\end{document}',1)[0]
    s=re.split(r'\\(?:bibliography|printbibliography|appendix)\b|\\begin\{thebibliography\}',s)[0]
    # Drop excluded sections until the next equal/higher-level section, conservatively.
    chunks=re.split(r'(\\(?:sub)*section\*?\s*\{[^}]*\})',s);keep=[];excluded=False
    for c in chunks:
        if re.match(r'\\(?:sub)*section',c):
            excluded=bool(re.search(r'references|bibliograph|acknowledg|limitation|ethic|broader|impact',c,re.I));continue
        if not excluded:keep.append(c)
    s='\n'.join(keep)
    for env in ['figure','table','equation','align','gather','multline','displaymath','tikzpicture','algorithm']:
        s=re.sub(r'\\begin\{'+env+r'\*?\}.*?\\end\{'+env+r'\*?\}',' ',s,flags=re.S)
    s=re.sub(r'\$\$.*?\$\$|\$(?:\\.|[^$])*\$|\\\[.*?\\\]|\\\(.*?\\\)',' ',s,flags=re.S)
    s=re.sub(r'\\(?:cite\w*|ref|eqref|label|url|includegraphics|caption|author|title|thanks|footnote|newcommand|renewcommand|def)\*?(?:\[[^\]]*\])?\s*\{[^{}]*\}',' ',s)
    s=s.replace('---','—').replace('~',' ').replace('``','“').replace("''",'”')
    s=re.sub(r'\\[A-Za-z@]+\*?(?:\[[^\]]*\])?',' ',s);s=re.sub(r'\\([^A-Za-z])',r'\1',s);s=s.replace('{','').replace('}','')
    paragraphs=[re.sub(r'\s+',' ',p).strip() for p in re.split(r'\n\s*\n',s)]
    return [p for p in paragraphs if len(p.split())>=40]
def parser_checks():
    for name,kind in [('../escape.tex',tarfile.REGTYPE),('/escape.tex',tarfile.REGTYPE),('link.tex',tarfile.SYMTYPE)]:
        b=io.BytesIO()
        with tarfile.open(fileobj=b,mode='w') as t:
            info=tarfile.TarInfo(name);info.type=kind;info.size=0;t.addfile(info,io.BytesIO())
        try:safe_tex(b.getvalue())
        except ValueError as e:assert str(e)=='unsafe_archive'
        else:raise AssertionError('unsafe archive accepted')
    assert prose('\\begin{document}\n'+('ordinary prose words '*50)+'\n\\bibliography{refs}\nsecret tail')
    assert not prose('\\begin{document}\n\\begin{figure}'+('caption '*100)+'\\end{figure}\n')

def main():
    parser_checks()
    p=argparse.ArgumentParser();p.add_argument('--workdir',type=pathlib.Path,required=True);p.add_argument('--list',type=pathlib.Path,required=True);p.add_argument('--audit',type=pathlib.Path,required=True);p.add_argument('--private',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('legacy_features',pathlib.Path(__file__).parents[1]/'features.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    entries={hashlib.sha256((r[0]+r[1]).encode()).hexdigest()[:12]:(r[0],r[1]) for r in csv.reader(a.list.open(),delimiter='\t')}
    audit=list(csv.DictReader(a.audit.open()));eligible=[r for r in audit if r['active_feature_key']=='True' and r['parse_status']=='parsed' and r['corpus'].startswith('iclr') and not (int(r['corpus'][-4:])<=2022 and r['after_2022_11_30']=='True')]
    selected=[]
    for era in ['pre','post']:
        pool=[r for r in eligible if (int(r['corpus'][-4:])<=2022 if era=='pre' else int(r['corpus'][-4:])>=2024)]
        selected.extend(dict(r,era=era) for r in sorted(pool,key=lambda r:r['paper_hash'])[:12])
    # Freeze selection before reading features or computing any outcome.
    a.out.parent.mkdir(parents=True,exist_ok=True);a.private.mkdir(parents=True,exist_ok=True)
    public_selection=[{k:r[k] for k in ['paper_hash','corpus','era','arxiv_version','stamp_date','pdf_sha256']} for r in selected]
    pathlib.Path(str(a.out)+'_selection.json').write_text(json.dumps(public_selection,indent=2)+'\n')
    def metrics(paras):
        text='\n'.join(paras);n=len(f.WORD.findall(text));sents=[s for p in paras for s in f.sentences(p)];_,core,func,_,_=f.ing_riders(sents)
        counts=[core,func,len(f.BOOST.findall(text)),sum(bool(f.OPENER_RE['additionally'].match(s)) for s in sents),sum(bool(f.OPENER_RE['notably'].match(s)) for s in sents),text.count('—'),sum(bool(f.THIS_CLOSER.match(s)) for s in sents),len(f.STYLE_LEX.findall(text)),len(f.FAMOUS.findall(text))]
        return n,{k:v*1000/n for k,v in zip(METRICS,counts)} if n else {}
    rows=[];failures=0;last=0
    for r in selected:
        corpus,ident=entries[r['paper_hash']];version=r['arxiv_version'];row={**{k:r[k] for k in ['paper_hash','corpus','era','arxiv_version']},'status':'not_attempted','http_status':'','source_sha256':'','tex_files':0,'included_tex_files':0,'pdf_words':0,'source_words':0}
        if failures>=3:rows.append(row);continue
        dest=a.private/(r['paper_hash']+'v'+version+'.source')
        receipt=dest.with_suffix('.retrieval.json')
        if receipt.exists():
            row.update(json.loads(receipt.read_text()));rows.append(row);continue
        try:
            if dest.exists():blob=dest.read_bytes();row['http_status']='cached'
            else:
                time.sleep(max(0,3-(time.monotonic()-last)));last=time.monotonic()
                with urllib.request.urlopen('https://arxiv.org/src/'+ident+'v'+version,timeout=35) as response:
                    row['http_status']=response.status;blob=response.read(30_000_001)
                if len(blob)>30_000_000:raise ValueError('oversize_archive')
                dest.write_bytes(blob)
            failures=0;row['source_sha256']=sha(blob)
        except Exception as e:
            failures+=1;row['status']=str(e) if isinstance(e,ValueError) else 'retrieval_failed';row['http_status']=row['http_status'] or getattr(e,'code',type(e).__name__);receipt.write_text(json.dumps({'status':row['status'],'http_status':row['http_status']}));rows.append(row);continue
        try:
            source,nfiles,nincluded=safe_tex(blob);row['tex_files']=nfiles;row['included_tex_files']=nincluded
            paras=prose(source);pdf=json.loads((a.workdir/'text'/corpus/(ident+'.json')).read_text());pn,pvals=metrics(f.drop_policy(pdf));sn,svals=metrics(paras);row.update(pdf_words=pn,source_words=sn)
            if min(pn,sn)<1500:row['status']='below_1500_words'
            else:
                row['status']='paired'
                for k in METRICS:row['pdf_'+k]=pvals[k];row['source_'+k]=svals[k];row['delta_'+k]=svals[k]-pvals[k]
        except Exception as e:row['status']=str(e) if isinstance(e,ValueError) else 'parse_error'
        rows.append(row)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with pathlib.Path(str(a.out)+'.csv').open('w') as out:
        w=csv.DictWriter(out,fields);w.writeheader();w.writerows(rows)
    summaries={}
    for era in ['pre','post']:
        paired=[r for r in rows if r['era']==era and r['status']=='paired'];summaries[era]={'selected':sum(r['era']==era for r in rows),'paired':len(paired),'mean_pdf_words':statistics.mean(r['pdf_words'] for r in paired) if paired else None,'mean_source_words':statistics.mean(r['source_words'] for r in paired) if paired else None,'mean_pdf':{k:statistics.mean(r['pdf_'+k] for r in paired) for k in METRICS} if paired else {},'mean_source':{k:statistics.mean(r['source_'+k] for r in paired) for k in METRICS} if paired else {},'mean_delta_source_minus_pdf':{k:statistics.mean(r['delta_'+k] for r in paired) for k in METRICS} if paired else {}}
    summary={'selection_rule':'first 12 lexicographic anonymous hashes per era; pre ICLR<=2022 excluding late stamp; post ICLR>=2024; no replacement','list_sha256':sha(a.list.read_bytes()),'audit_sha256':sha(a.audit.read_bytes()),'legacy_features_sha256':sha(pathlib.Path(f.__file__).read_bytes()),'selection_sha256':sha(pathlib.Path(str(a.out)+'_selection.json').read_bytes()),'parser_checks':5,'statuses':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},'eras':summaries}
    pathlib.Path(str(a.out)+'.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
