#!/usr/bin/env python3
"""Frozen-feature statistical sensitivity audit. No raw text, identifiers, or model calls."""
import argparse, ast, gzip, hashlib, importlib.metadata, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 20261019

def literals(path, names):
    tree = ast.parse(path.read_text())
    return {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(src, scripts, features=None):
    definitions = literals(scripts/'analyze.py', ['VENUES', 'SERIES'])
    fs = literals(scripts/'stats.py', ['STYLE', 'INDEX9'])
    fp=features or src/'features.jsonl'
    opener=gzip.open if str(fp).endswith('.gz') else open
    with opener(fp,'rt') as stream: rows = [json.loads(s) for s in stream]
    for r in rows:
        if 'paper_hash' not in r: r['paper_hash']=hashlib.sha256((r['corpus']+str(r['id'])).encode()).hexdigest()[:12]
    d = pd.DataFrame([{k:v for k,v in r.items() if k not in ('fw','id','intro_first','intro_open')} for r in rows])
    d['year'] = d.corpus.str.extract(r'(\d{4})')[0].astype(int)
    mapping = {}
    for v, (ctrl, pre, post) in definitions['VENUES'].items():
        if v.endswith('*'): continue
        for cs, era in [(ctrl,'2016'), (pre,'pre'), (post,'post')]:
            mapping.update({c:(v,era) for c in cs})
    for v, cs in definitions['SERIES'].items():
        mapping.update({c:(v,'series') for c in cs[1:-1]})
        for y in (2017,2018,2019):
            c = cs[0][:-4]+str(y)
            if c != 'mlsys2019': mapping[c]=(v,'trend')
    d['venue'] = d.corpus.map(lambda c:mapping.get(c,(None,None))[0])
    d['era'] = d.corpus.map(lambda c:mapping.get(c,(None,None))[1])
    original = d[d.venue.notna()].copy()
    corrected = d.copy()
    corrected.loc[corrected.corpus=='vldb2016',['venue','era']]=['VLDB','2016']
    corrected = corrected[corrected.venue.notna()].copy()
    assert np.isfinite(corrected[fs['STYLE']].to_numpy(float)).all(), 'Base STYLE missing/nonfinite values require explicit complete-case handling'
    return d, original, corrected, fs

def transform(train, test, fs, mode='venue', sd='pre'):
    """All estimated quantities from train. Era labels in train select historical baseline."""
    x = test[fs].to_numpy(float).copy()
    baseline = train[train.era=='pre']
    for v in test.venue.unique():
        m = (test.venue==v).to_numpy()
        b = baseline if mode=='pooled' else baseline[baseline.venue==v]
        s = train if sd=='pooled' else b
        if mode!='pooled' and sd=='pooled': s=train[train.venue==v]
        if not len(b): raise ValueError(f'No training baseline for {v}')
        mean=b[fs].mean().to_numpy(float)
        scale=s[fs].std(ddof=0).to_numpy(float)
        scale=np.where((scale==0)|~np.isfinite(scale),1.,scale)
        x[m]=(x[m]-mean)/scale
    return np.clip(x,-8,8)

def fitauc(x,y,xt,yt):
    clf=LogisticRegression(C=.3,max_iter=2000)
    clf.fit(x,y)
    return float(roc_auc_score(yt,clf.predict_proba(xt)[:,1]))

def classifiers(d, fs):
    d=d[d.era.isin(['pre','post'])].copy().reset_index(drop=True)
    rows=[]
    featuresets={'style33':fs['STYLE'], 'syntax_stance31':[f for f in fs['STYLE'] if f not in ['style_lex_1k','famous_ai_words_1k']], 'index9':fs['INDEX9']}
    for name, features in featuresets.items():
        assert np.isfinite(d[features].to_numpy(float)).all()
        y=(d.era=='post').to_numpy(int)
        legacy=transform(d,d,features)
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=0).split(d,y)):
            tr,te=d.iloc[a],d.iloc[b]
            for mode in ['legacy_full_data_venue_pre','strict_fold_venue_pre','strict_fold_pooled_pre','strict_fold_venue_pooled_sd']:
                if mode.startswith('legacy'): x,xt=legacy[a],legacy[b]
                else:
                    group='pooled' if mode=='strict_fold_pooled_pre' else 'venue'
                    sd='pooled' if mode.endswith('pooled_sd') else 'pre'
                    x,xt=transform(tr,tr,features,group,sd),transform(tr,te,features,group,sd)
                rows.append(dict(feature_set=name,scheme=mode,fold=str(fold),n_train=len(tr),n_test=len(te),auc=fitauc(x,y[a],xt,y[b])))
        for v in sorted(d.venue.unique()):
            tr=d[d.venue!=v]; te=d[d.venue==v]
            yt=(tr.era=='post').to_numpy(int); ye=(te.era=='post').to_numpy(int)
            # Original LOVO deliberately lets the target venue's entire historical baseline normalize its test rows.
            rows.append(dict(feature_set=name,scheme='legacy_lovo_target_pre_includes_evaluation',fold=v,n_train=len(tr),n_test=len(te),auc=fitauc(legacy[d.venue!=v],yt,legacy[d.venue==v],ye)))
            # No target venue observations used for scaling or fitting.
            rows.append(dict(feature_set=name,scheme='strict_lovo_unseen_venue_pooled_pre',fold=v,n_train=len(tr),n_test=len(te),auc=fitauc(transform(tr,tr,features,'pooled'),yt,transform(tr,te,features,'pooled'),ye)))
            # Historical target baseline availability is a different deployment contract.
            p=te[te.era=='pre']; rng=np.random.default_rng(SEED)
            cal=p.iloc[rng.permutation(len(p))[:len(p)//2]]
            evalte=te.drop(cal.index)
            calibration=pd.concat([tr,cal])
            rows.append(dict(feature_set=name,scheme='lovo_disjoint_target_pre_calibration',fold=v,n_train=len(tr),n_calibration=len(cal),n_test=len(evalte),auc=fitauc(transform(tr,tr,features),yt,transform(calibration,evalte,features),(evalte.era=='post').to_numpy(int))))
    return pd.DataFrame(rows)

def historical_scores(d,fs):
    rows=[]
    tr=d[d.era.isin(['pre','post'])]
    te=d[~d.era.isin(['pre','post'])]
    for name,features in [('style33',fs['STYLE']),('syntax_stance31',[f for f in fs['STYLE'] if f not in ['style_lex_1k','famous_ai_words_1k']]),('index9',fs['INDEX9'])]:
        clf=LogisticRegression(C=.3,max_iter=2000).fit(transform(tr,tr,features),(tr.era=='post').to_numpy(int))
        for mode in ['venue','pooled']:
            # Refit for matching training representation; no evaluation rows determine scaling.
            clf.fit(transform(tr,tr,features,mode),(tr.era=='post').to_numpy(int))
            scored=te[['venue','year','era']].copy()
            scored['score']=clf.predict_proba(transform(tr,te,features,mode))[:,1]
            for (v,y,e),g in scored.groupby(['venue','year','era']):
                rows.append(dict(feature_set=name,normalization=mode,venue=v,year=y,era=e,n=len(g),mean_post_era_score=g.score.mean(),share_score_gt_half=(g.score>.5).mean()))
    return pd.DataFrame(rows)

def index_values(d, features, scaling):
    x=pd.Series(index=d.index,dtype=float)
    for v,g in d.groupby('venue'):
        pre=g[g.era=='pre'][features]
        denom=g[g.era.isin(['pre','post'])][features] if scaling.startswith('pooled_sd') else pre
        scale=denom.std(ddof=0).clip(lower=.001)
        x.loc[g.index]=((g[features]-pre.mean())/scale).mean(axis=1)
    if scaling.endswith('composite_pre_sd'):
        for v,g in d.groupby('venue'):
            baseline=x.loc[g[g.era=='pre'].index]
            x.loc[g.index]=(x.loc[g.index]-baseline.mean())/max(baseline.std(ddof=0),.001)
    return x

def cells_for(d,values):
    return d.assign(value=values).groupby(['venue','year','era']).agg(value=('value','mean'),n=('value','size')).reset_index()

def design(c, trend):
    venues=sorted(c.venue.unique()); v=np.column_stack([(c.venue==a).to_numpy(float) for a in venues])
    post=(c.year>=2024).to_numpy(float)
    return np.column_stack([v,v*(c.year.to_numpy()-2020)[:,None],post]) if trend else np.column_stack([v,post])

def shift(c,weight='cell',trend=True):
    c=c[((c.year<=2022)|(c.year>=2024))] if trend else c[c.era.isin(['pre','post'])]
    x=design(c,trend); w=np.sqrt(c.n.to_numpy() if weight=='paper' else np.ones(len(c)))
    return float(np.linalg.lstsq(x*w[:,None],c.value.to_numpy()*w,rcond=None)[0][-1])

def shift_bootstrap(c,weight,trend,B,rng):
    # Frisch-Waugh-Lovell: nuisance intercept/trend is venue-specific, so whole-venue
    # bootstrap is exactly a resampling of within-venue cross products.
    c=c[((c.year<=2022)|(c.year>=2024))] if trend else c[c.era.isin(['pre','post'])]
    components=[]
    for _,g in c.groupby('venue'):
        w=np.sqrt(g.n.to_numpy() if weight=='paper' else np.ones(len(g)))
        z=np.column_stack([np.ones(len(g)),g.year.to_numpy()-2020]) if trend else np.ones((len(g),1))
        z=z*w[:,None];y=g.value.to_numpy()*w;p=(g.year>=2024).to_numpy(float)*w
        yr=y-z@np.linalg.lstsq(z,y,rcond=None)[0];pr=p-z@np.linalg.lstsq(z,p,rcond=None)[0]
        components.append((pr@yr,pr@pr))
    arr=np.array(components);draws=rng.integers(len(arr),size=(B,len(arr)))
    return arr[draws,0].sum(axis=1)/arr[draws,1].sum(axis=1)

def venue_resample(c,rng):
    vs=sorted(c.venue.unique()); out=[]
    for i,v in enumerate(rng.choice(vs,len(vs),replace=True)):
        a=c[c.venue==v].copy();a['venue']=str(i);out.append(a)
    return pd.concat(out,ignore_index=True)

def bicfit(y,x):
    rss=np.square(y-x@np.linalg.lstsq(x,y,rcond=None)[0]).sum()
    return float(len(y)*np.log(max(rss/len(y),1e-30))+np.linalg.matrix_rank(x)*np.log(len(y)))

def cp(c,candidates):
    c=c[c.year<=2025];vs=sorted(c.venue.unique());t=c.year.to_numpy()-2020
    x=np.column_stack([np.ones(len(c)), *[(c.venue==v).to_numpy(float) for v in vs[1:]], t])
    y=c.value.to_numpy();base=bicfit(y,x)
    vals={y0:bicfit(y,np.column_stack([x,(c.year>=y0).to_numpy(float)])) for y0 in candidates}
    best=min(vals,key=vals.get)
    return int(best),float(base-vals[best]),vals

def inference(d,fs,B):
    rows=[];loo=[];cp_rows=[];cpboot=[]
    subsets={'index9':fs['INDEX9'],'index7_no_lexicons':[f for f in fs['INDEX9'] if f not in ['style_lex_1k','famous_ai_words_1k']], 'index6_no_lexicons_emdash':[f for f in fs['INDEX9'] if f not in ['style_lex_1k','famous_ai_words_1k','emdash_1k']]}
    for setname,features in subsets.items():
        for scaling in ['pooled_sd','pre_sd','pooled_sd_then_composite_pre_sd']:
            values=index_values(d,features,scaling);c=cells_for(d,values)
            for weight in ['paper','cell']:
                for trend in [False,True]:
                    rng=np.random.default_rng(SEED);est=shift(c,weight,trend)
                    boots=shift_bootstrap(c,weight,trend,B,rng)
                    fitc=c[((c.year<=2022)|(c.year>=2024))] if trend else c[c.era.isin(['pre','post'])]
                    rows.append(dict(feature_set=setname,scaling=scaling,weight=weight,trend_adjusted=trend,estimate=est,venue_boot_lo=float(np.quantile(boots,.025)),venue_boot_hi=float(np.quantile(boots,.975)),bootstrap_B=B,n_venues=c.venue.nunique(),n_cells=len(fitc),n_papers=int(fitc.n.sum())))
                    for v in sorted(c.venue.unique()):
                        loo.append(dict(feature_set=setname,scaling=scaling,weight=weight,trend_adjusted=trend,omitted=v,estimate=shift(c[c.venue!=v],weight,trend)))
            if setname!='index9':continue
            for selection in ['all10','acl_icml']:
                cc=c[c.venue.isin(['ACL','ICML'])].copy() if selection=='acl_icml' else c.copy()
                cc=cc[cc.year<=2025].copy()
                # Corrected VLDB2016 and all legacy mapped trend/control/series cells are retained.
                for window,candidates in [('legacy',list(range(2021,2026))),('two_years_each_side_overall',list(range(2021,2025)))]:
                    best,delta,bics=cp(cc,candidates)
                    cp_rows.append(dict(scaling=scaling,selection=selection,window=window,best_tau=best,delta_bic_vs_linear=delta,n_cells=len(cc),n_venues=cc.venue.nunique(),**{f'bic_{k}':v for k,v in bics.items()}))
                    rng=np.random.default_rng(SEED)
                    # Entire venue trajectories preserve serial dependence, but only two clusters for ACL/ICML.
                    choices=[cp(venue_resample(cc,rng),candidates)[0] for _ in range(B)]
                    cpboot.append(dict(scaling=scaling,selection=selection,window=window,resampling='venue_trajectories',B=B,**{f'p_tau_{k}':choices.count(k)/B for k in candidates}))
                    # Conditional sampling of papers within fixed cells does NOT measure venue uncertainty.
                    dd=d[d.venue.isin(cc.venue.unique()) & (d.year<=2025)].copy();dd['value']=values.loc[dd.index]
                    groups=[g.value.to_numpy() for _,g in dd.groupby(['venue','year','era'])]
                    template=cells_for(dd,dd.value)
                    choices=[]
                    for _ in range(B):
                        cb=template.copy();cb['value']=[rng.choice(a,len(a),replace=True).mean() for a in groups]
                        choices.append(cp(cb,candidates)[0])
                    cpboot.append(dict(scaling=scaling,selection=selection,window=window,resampling='papers_within_fixed_cells_fixed_scaling',B=B,**{f'p_tau_{k}':choices.count(k)/B for k in candidates}))
                    for v in sorted(cc.venue.unique()):
                        b,de,_=cp(cc[cc.venue!=v],candidates)
                        loo.append(dict(feature_set='changepoint_'+selection,scaling=scaling,weight='cell',trend_adjusted=window,omitted=v,estimate=b,delta_bic_vs_linear=de))
    return [pd.DataFrame(a) for a in [rows,loo,cp_rows,cpboot]]

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--features',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=500);a=p.parse_args()
    scripts=Path(__file__).resolve().parents[1];a.out.mkdir(parents=True,exist_ok=True)
    if a.source is None and a.features is None: p.error('supply --features or --source')
    raw,original,corrected,fs=load(a.source,scripts,a.features)
    hashes={name:sha(a.source/name) for name in ['features.jsonl','newfeatures.jsonl','all_features.csv']} if a.source else {}
    if a.features: hashes['portable_features']=sha(a.features)
    manifest=dict(seed=SEED,bootstrap_B=a.bootstrap,source_sha256=hashes,definition_sha256={n:sha(scripts/n) for n in ['stats.py','analyze.py']},packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','statsmodels','scikit-learn']},python=platform.python_version(),definitions=fs,counts={'base_rows':len(raw),'base_cells':raw.corpus.nunique(),'original_rows':len(original),'original_cells':original.corpus.nunique(),'corrected_rows':len(corrected),'corrected_cells':corrected.corpus.nunique(),'excluded_original':raw[raw.venue.isna()].groupby('corpus').size().to_dict()})
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    audit_path=scripts.parent/'results/corpus_validation/arxiv_versions.csv'
    audit=pd.read_csv(audit_path)
    dated=audit[(audit.active_feature_key.astype(str).str.lower()=='true') & (audit.after_2022_11_30.astype(str).str.lower()=='true')]
    exclude=set(dated.paper_hash)
    dateclean=corrected[~((corrected.year<=2022)&corrected.paper_hash.isin(exclude))]
    manifest['version_audit_sha256']=sha(audit_path)
    manifest['version_exclusions']=len(corrected)-len(dateclean)
    assert manifest['version_exclusions']==18, manifest['version_exclusions']
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    counts=[]
    for cohort,d in [('legacy_mapping',original),('corrected_vldb2016',corrected),('legacy_mapping_no_iclr',original[original.venue!='ICLR']),('corrected_vldb2016_no_iclr',corrected[corrected.venue!='ICLR']),('corrected_no_late_arxiv',dateclean)]:
        count=d.groupby(['venue','era','year']).size().rename('n').reset_index();count.insert(0,'cohort',cohort);counts.append(count)
        print(cohort,len(d),flush=True)
        hs=historical_scores(d,fs);hs.insert(0,'cohort',cohort);hs.to_csv(a.out/f'{cohort}_historical_scores.csv',index=False)
        cl=classifiers(d,fs);cl.insert(0,'cohort',cohort);cl.to_csv(a.out/f'{cohort}_classifier.csv',index=False)
        cl.groupby(['feature_set','scheme']).agg(mean_auc=('auc','mean'),sd_auc=('auc','std'),folds=('auc','size')).reset_index().to_csv(a.out/f'{cohort}_classifier_summary.csv',index=False)
        for name,frame in zip(['shift','leave_one_venue_out','changepoint','changepoint_bootstrap'],inference(d,fs,a.bootstrap)):
            frame.insert(0,'cohort',cohort);frame.to_csv(a.out/f'{cohort}_{name}.csv',index=False)
    pd.concat(counts).to_csv(a.out/'cohort_counts.csv',index=False)
    print('Complete',flush=True)
if __name__=='__main__':main()
