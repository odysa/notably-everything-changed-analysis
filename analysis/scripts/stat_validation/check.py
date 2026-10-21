#!/usr/bin/env python3
"""Contract checks independent of reported empirical effect sizes."""
import importlib.util
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
spec=importlib.util.spec_from_file_location('validate',Path(__file__).with_name('validate.py'))
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
# Changing an evaluation observation cannot change training-derived normalization.
tr=pd.DataFrame({'venue':['A']*4,'era':['pre','pre','post','post'],'f':[0.,2.,4.,6.]})
te=pd.DataFrame({'venue':['A','A'],'era':['pre','post'],'f':[1.,9.]})
x=v.transform(tr,te,['f']);other=te.copy();other.loc[1,'f']=1e12
assert np.array_equal(x[0],v.transform(tr,other,['f'])[0])
assert v.transform(tr,other,['f'])[1,0]==8
# Unseen-venue transform is independent of target baseline labels.
u=te.assign(venue='NEW'); assert np.array_equal(v.transform(tr,u,['f'],'pooled'),v.transform(tr,u.assign(era='post'),['f'],'pooled'))
# Exact hand-computed within-venue contrast and weighted-vs-unweighted behavior.
c=pd.DataFrame({'venue':['A','A','B','B'],'year':[2020,2025,2020,2025],'era':['pre','post','pre','post'],'value':[0.,2.,0.,4.],'n':[100,100,10,10]})
assert abs(v.shift(c,'cell',False)-3.)<1e-10
assert abs(v.shift(c,'paper',False)-24/11)<1e-10
# Fixed candidate universe; planted step in exact cell-level data recovered.
c=pd.DataFrame({'venue':np.repeat(['A','B'],10),'year':list(range(2016,2026))*2})
c['value']=(c.year>=2023).astype(float)+(c.year-2020)*.01
assert v.cp(c,list(range(2021,2025)))[0]==2023
print('5 contracts passed: heldout independence, clipping, unseen labels, weighting, planted changepoint')
# Algebraic venue-bootstrap acceleration must match explicit cluster resampling.
c=pd.DataFrame({'venue':np.repeat(['A','B','C'],6),'year':list(range(2019,2025))*3})
c['era']=np.where(c.year==2020,'pre',np.where(c.year>=2024,'post','series'))
c['n']=np.arange(1,len(c)+1)
c['value']=.03*(c.year-2020)+(c.year>=2024)*(.5+np.arange(len(c))*.01)
r1=np.random.default_rng(55);r2=np.random.default_rng(55)
a=v.shift_bootstrap(c,'paper',True,20,r1)
b=np.array([v.shift(v.venue_resample(c,r2),'paper',True) for _ in range(20)])
assert np.allclose(a,b,atol=1e-10)
print('Accelerated whole-venue bootstrap equivalence passed')
# Frozen-data regression reference independently reproduced with legacy stats.part_b.
scripts=Path(__file__).resolve().parents[1]
portable=scripts.parent/'results/corpus_validation/features_numeric.jsonl.gz'
if portable.exists():
    raw,d,corrected,fs=v.load(None,scripts,portable)
    assert (len(raw),len(d),len(corrected))==(9472,9023,9101)
    assert (raw.corpus.nunique(),d.corpus.nunique(),corrected.corpus.nunique())==(103,98,99)
    cells=v.cells_for(d,v.index_values(d,fs['INDEX9'],'pooled_sd_then_composite_pre_sd'))
    assert abs(v.shift(cells,'paper',True)-1.9669085156749961)<1e-10
    print('Frozen cohort counts and legacy part_b coefficient reproduction passed')
