#!/usr/bin/env python3
"""How well can a style-only classifier tell pre-LLM from post-GPT-4 papers?

Usage: python detect_eval.py <workdir> <outdir>
Trains L2 logistic regression on 33 style features (no topic words, no structure), z-scored per
venue against that venue's 2019-20 papers, on 2019-20 (label 0) vs 2024-26 (label 1) papers of the
eight main venues. Reports:
  - 5-fold cross-validated accuracy, balanced accuracy, AUC, and detection rate (TPR) at fixed
    false-positive rates (thresholds set on the cross-validated pre-LLM scores);
  - leave-one-venue-out versions of the same;
  - with the final model: false-positive rate on held-out pre-ChatGPT papers (2016, 2018, 2021,
    2022), share flagged in 2023 and 2024, and detection rate on unseen post-GPT-4 papers
    (MLSys/OSDI/NSDI 2026 are in training; COLM 2025 and EuroSys 2025-26 are unseen venues).
Writes detect_eval.json and detect_eval.md.
"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats as S  # noqa: E402

wd, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(os.path.join(wd, "all_features.csv"))
feats = S.STYLE
# Per-venue z-scores against the venue's own 2019-20 papers (COLM/EuroSys use pooled 2020 stats).
pre_all = df[(df.era == "pre") & df.venue.isin(S.MAIN)]
X = df[feats].astype(float).copy()
for v in df.venue.dropna().unique():
    m = df.venue == v
    base = df[m & (df.era == "pre")]
    if len(base) < 20:
        base = pre_all
    for f in feats:
        X.loc[m, f] = (X.loc[m, f] - base[f].mean()) / (base[f].std(ddof=0) or 1.0)
X = X.clip(-8, 8).fillna(0)
train = df.venue.isin(S.MAIN) & df.era.isin(["pre", "post"])
y = (df.era == "post").astype(int)
clf = LogisticRegression(C=0.3, max_iter=3000)


def summarize(yt, p, thr_scores=None):
    yt, p = np.asarray(yt), np.asarray(p)
    pre_scores = p[yt == 0] if thr_scores is None else thr_scores
    res = {"n": int(len(yt)), "n_pre": int((yt == 0).sum()), "n_post": int((yt == 1).sum()),
           "auc": float(roc_auc_score(yt, p)),
           "accuracy@0.5": float(((p > 0.5) == yt).mean()),
           "balanced_accuracy@0.5": float(0.5 * ((p[yt == 1] > 0.5).mean() + (p[yt == 0] <= 0.5).mean()))}
    for fpr in (0.01, 0.05, 0.10):
        t = float(np.quantile(pre_scores, 1 - fpr))
        res[f"tpr@fpr{int(fpr * 100)}"] = float((p[yt == 1] > t).mean())
        res[f"thr@fpr{int(fpr * 100)}"] = t
    return res


# 5-fold CV within venues
cvp = cross_val_predict(clf, X[train], y[train], cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
cv = summarize(y[train], cvp)
# Leave-one-venue-out
lovo = {}
for v in S.MAIN:
    tr = train & (df.venue != v)
    te = train & (df.venue == v)
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    lovo[v] = summarize(y[te], p)
# Final model on all training papers; thresholds from CV pre-LLM scores
clf.fit(X[train], y[train])
thr = {k: cv[f"thr@fpr{k}"] for k in (1, 5, 10)}
p_all = clf.predict_proba(X)[:, 1]
df["p"] = p_all
held = []
groups = [("2016 (pre-ChatGPT, unseen)", df.year == 2016), ("2017 (pre-ChatGPT, unseen)", df.year == 2017),
          ("2018 (pre-ChatGPT, unseen)", df.year == 2018), ("2019 (pre-ChatGPT, unseen)", (df.year == 2019) & (df.era == "trend")),
          ("2021 (pre-ChatGPT, unseen)", df.year == 2021), ("2022 (pre-ChatGPT, unseen)", df.year == 2022),
          ("2023 (transition, unseen)", df.year == 2023), ("2024 yearly samples (unseen)", (df.year == 2024) & (df.era == "series")),
          ("COLM 2025 (unseen venue)", df.corpus == "colm2025"), ("EuroSys 2025-26 (unseen venue)", df.corpus == "eurosys2025")]
base_df = pd.read_json(os.path.join(wd, "features.jsonl"), lines=True)[["corpus", "id"]]
for name, m in groups:
    m = m & df.p.notna()
    if m.sum() == 0:
        continue
    row = {"group": name, "n": int(m.sum()), "mean_p": float(df[m].p.mean()), "flagged@0.5": float((df[m].p > 0.5).mean())}
    for k in (1, 5, 10):
        row[f"flagged@fpr{k}"] = float((df[m].p > thr[k]).mean())
    held.append(row)
# COLM/EuroSys are absent from all_features (stats.load keeps the 8 venues); score them separately
extra = []
try:
    feats_all = pd.read_json(os.path.join(wd, "features.jsonl"), lines=True)
    nf = pd.read_json(os.path.join(wd, "newfeatures.jsonl"), lines=True)
    ex = feats_all[feats_all.corpus.isin(["colm2024", "colm2025", "colm2026", "eurosys2025"])].merge(nf, on=["corpus", "id"], how="left")
    if len(ex):
        Xe = ex[feats].astype(float).copy()
        for f in feats:
            Xe[f] = (Xe[f] - pre_all[f].mean()) / (pre_all[f].std(ddof=0) or 1.0)
        pe = clf.predict_proba(Xe.clip(-8, 8).fillna(0))[:, 1]
        for c in ["colm2024", "colm2025", "colm2026", "eurosys2025"]:
            m = (ex.corpus == c).values
            if m.sum():
                row = {"group": {"colm2024": "COLM 2024 (unseen venue)", "colm2025": "COLM 2025 (unseen venue)", "colm2026": "COLM 2026 (unseen venue)",
                                 "eurosys2025": "EuroSys 2025-26 (unseen venue)"}[c],
                       "n": int(m.sum()), "mean_p": float(pe[m].mean()), "flagged@0.5": float((pe[m] > 0.5).mean())}
                for k in (1, 5, 10):
                    row[f"flagged@fpr{k}"] = float((pe[m] > thr[k]).mean())
                extra.append(row)
except Exception as e:  # noqa: BLE001
    print("extra venues skipped:", e)
held = [h for h in held if h["n"] > 0 and "unseen venue" not in h["group"]] + extra
res = {"features": len(feats), "cv": cv, "lovo": lovo, "held_out": held}
# Score distributions for the website: cross-validated scores of the training papers and
# final-model scores of every held-out group (rounded; no paper ids).
yt = np.asarray(y[train])
scores = {"cv_pre": sorted(round(float(x), 3) for x in cvp[yt == 0]),
          "cv_post": sorted(round(float(x), 3) for x in cvp[yt == 1])}
for name, m in groups:
    m = m & df.p.notna()
    if m.sum() and "unseen venue" not in name:
        scores[name] = sorted(round(float(x), 3) for x in df[m].p)
try:
    for c, name in [("colm2024", "COLM 2024 (unseen venue)"), ("colm2025", "COLM 2025 (unseen venue)"),
                    ("colm2026", "COLM 2026 (unseen venue)"), ("eurosys2025", "EuroSys 2025-26 (unseen venue)")]:
        mm = (ex.corpus == c).values
        if mm.sum():
            scores[name] = sorted(round(float(x), 3) for x in pe[mm])
except NameError:
    pass
json.dump(scores, open(os.path.join(out, "detect_scores.json"), "w"))
json.dump(res, open(os.path.join(out, "detect_eval.json"), "w"), indent=1)

pct = lambda x: f"{x * 100:.0f}%"
md = ["| Setting | Papers | AUC | Accuracy (p>0.5) | Balanced acc. | Detected at 1% / 5% / 10% false alarms |", "|---|---|---|---|---|---|"]
md.append(f"| 5-fold CV, 8 venues | {cv['n']} | {cv['auc']:.3f} | {pct(cv['accuracy@0.5'])} | {pct(cv['balanced_accuracy@0.5'])} | "
          f"{pct(cv['tpr@fpr1'])} / {pct(cv['tpr@fpr5'])} / {pct(cv['tpr@fpr10'])} |")
for v, r in lovo.items():
    md.append(f"| held-out venue: {v} | {r['n']} | {r['auc']:.3f} | {pct(r['accuracy@0.5'])} | {pct(r['balanced_accuracy@0.5'])} | "
              f"{pct(r['tpr@fpr1'])} / {pct(r['tpr@fpr5'])} / {pct(r['tpr@fpr10'])} |")
md.append("")
md.append("| Held-out group | Papers | Flagged at p>0.5 | Flagged at 1% / 5% / 10% FPR thresholds |")
md.append("|---|---|---|---|")
for h in held:
    md.append(f"| {h['group']} | {h['n']} | {pct(h['flagged@0.5'])} | {pct(h['flagged@fpr1'])} / {pct(h['flagged@fpr5'])} / {pct(h['flagged@fpr10'])} |")
open(os.path.join(out, "detect_eval.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
