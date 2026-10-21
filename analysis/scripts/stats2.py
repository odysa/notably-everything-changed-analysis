#!/usr/bin/env python3
"""Robustness and placebo checks on top of stats.py / newfeat.py.

Usage: python stats2.py <workdir> <outdir>
Needs <workdir>/all_features.csv (from newfeat.py). Writes <outdir>/stats2_*.csv and stats2_summary.json.

  P  placebo: the same tests on 2016 vs 2019-20 (no AI existed). Pooled AUC of the style index and
     a style classifier trained to separate 2016 from 2020, vs the real 2020 vs 2024-26 versions
  R  regression with controls: style ~ post + log(words) + pages + log(refs) + LLM mentions + venue,
     HC3 robust SEs; also restricted to papers that never mention LLMs
  Q  quantile regression of the style index on post (+ venue) at the 10th, 25th, 50th, 75th, 90th pct
  E  equivalence (TOST) for features we call unchanged: is the shift inside +-0.2 pre-AI SD?
  D  dose-response within the post era: style ~ log(1 + LLM mentions per 1k) + venue
  T  sensitivity of the non-LLM control to the threshold (0, <1, <5 mentions)
  C  classifier permutation test: AUC with shuffled era labels (within venue), 200 runs
  L  leave-one-venue-out range of the pooled style-index AUC
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats as S  # noqa: E402

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(1)
wd, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(os.path.join(wd, "all_features.csv")).copy()
MAIN = S.MAIN
main = df[df.venue.isin(MAIN)].copy()
summary = {}


def pooled_auc(d, a_era, b_era, f, venues=MAIN):
    vals = []
    for v in venues:
        a = d[(d.venue == v) & (d.era == a_era)][f].dropna().values
        b = d[(d.venue == v) & (d.era == b_era)][f].dropna().values
        if len(a) >= 10 and len(b) >= 10:
            vals.append(S.auc(a, b))
    return float(np.mean(vals)), vals


# ---- P: placebo ----
ctrl_venues = [v for v in MAIN if (main[(main.venue == v) & (main.era == "2016")].shape[0] > 0)]
rows = []
for f in ["style_index", "ing_func_1k", "ing_interp_core_1k", "op_additionally_1k", "boost_1k", "pct_long_words", "flesch"]:
    real, _ = pooled_auc(main, "pre", "post", f, ctrl_venues)
    plac, _ = pooled_auc(main, "2016", "pre", f, ctrl_venues)
    rows.append({"feature": f, "placebo_auc_2016_vs_2020": plac, "real_auc_2020_vs_post": real})


def clf_auc(d, a_era, b_era, feats):
    d = d[d.era.isin([a_era, b_era])].copy()
    X = S.zfeat(d.assign(era=d.era.replace({a_era: "pre"})), feats)
    y = (d.era == b_era).astype(int).values
    p = cross_val_predict(LogisticRegression(C=0.3, max_iter=2000), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


cv_ = main[main.venue.isin(ctrl_venues)]
plac_clf = clf_auc(cv_, "2016", "pre", S.STYLE)
real_clf = clf_auc(cv_, "pre", "post", S.STYLE)
rows.append({"feature": "style classifier (5-fold CV)", "placebo_auc_2016_vs_2020": plac_clf, "real_auc_2020_vs_post": real_clf})
pd.DataFrame(rows).to_csv(os.path.join(out, "stats2_placebo.csv"), index=False)
summary["placebo"] = rows
print("P", pd.DataFrame(rows).round(3).to_string())

# ---- R: regression with controls ----
d = main[main.era.isin(["pre", "post"])].copy()
d["post"] = (d.era == "post").astype(int)
d["log_words"] = np.log(d.words)
d["log_refs"] = np.log1p(d.n_refs)
d["log_llm"] = np.log1p(d.llm_1k)
rows = []
for f in ["style_index", "ing_func_1k", "ing_interp_core_1k", "op_additionally_1k", "boost_1k", "hedge_1k", "we_1k", "pct_long_words", "flesch"]:
    # Standardize within venue by that venue's pre-LLM SD (same scale as stats.py).
    for v in MAIN:
        m = d.venue == v
        sd = d[m & (d.era == "pre")][f].std(ddof=0) or 1.0
        d.loc[m, "y"] = d.loc[m, f] / sd
    m0 = smf.ols("y ~ post + C(venue)", data=d).fit(cov_type="HC3")
    m1 = smf.ols("y ~ post + log_words + pages + log_refs + log_llm + C(venue)", data=d).fit(cov_type="HC3")
    m2 = smf.ols("y ~ post + log_words + pages + log_refs + C(venue)", data=d[(d.llm_1k == 0)]).fit(cov_type="HC3")
    rows.append({"feature": f, "post_raw_sd": m0.params["post"], "post_controlled_sd": m1.params["post"], "se_controlled": m1.bse["post"],
                 "p_controlled": m1.pvalues["post"], "post_zero_llm_papers_sd": m2.params["post"], "se_zero_llm": m2.bse["post"],
                 "n_zero_llm_post": int(((d.llm_1k == 0) & (d.post == 1)).sum()), "r2_controlled": m1.rsquared})
pd.DataFrame(rows).to_csv(os.path.join(out, "stats2_controls.csv"), index=False)
summary["controls"] = rows
print("R", pd.DataFrame(rows).round(3).to_string())

# ---- Q: quantile regression ----
rows = []
for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
    m = smf.quantreg("style_index ~ post + C(venue)", data=d).fit(q=q)
    ci = m.conf_int().loc["post"].values
    rows.append({"quantile": q, "post_shift": m.params["post"], "ci_lo": ci[0], "ci_hi": ci[1]})
pd.DataFrame(rows).to_csv(os.path.join(out, "stats2_quantile.csv"), index=False)
summary["quantile"] = rows
print("Q", pd.DataFrame(rows).round(3).to_string())

# ---- E: equivalence (TOST) on standardized within-venue differences ----
rows = []
for f in ["sent_len", "sent_cv", "not_only_1k", "triad_1k", "semicolon_1k", "question_1k", "first_claim_1k", "n_sections", "intro_words",
          "related_words", "n_figs", "ing_func_1k"]:
    if f not in d:
        continue
    diffs, ses = [], []
    for v in MAIN:
        a = d[(d.venue == v) & (d.era == "pre")][f].values
        b = d[(d.venue == v) & (d.era == "post")][f].values
        sd = a.std(ddof=0) or 1.0
        diffs.append((b.mean() - a.mean()) / sd)
        ses.append(np.sqrt(b.var(ddof=1) / len(b) + a.var(ddof=1) / len(a)) / sd)
    w = 1 / np.square(ses)
    est = float((w * diffs).sum() / w.sum())
    se = float(np.sqrt(1 / w.sum()))
    margin = 0.2
    p_lo = 1 - st.norm.cdf((est + margin) / se)
    p_hi = st.norm.cdf((est - margin) / se)
    rows.append({"feature": f, "shift_sd": est, "ci90_lo": est - 1.645 * se, "ci90_hi": est + 1.645 * se,
                 "tost_p": max(p_lo, p_hi), "equivalent_within_0.2sd": bool(max(p_lo, p_hi) < 0.05)})
pd.DataFrame(rows).to_csv(os.path.join(out, "stats2_equivalence.csv"), index=False)
summary["equivalence"] = rows
print("E", pd.DataFrame(rows).round(3).to_string())

# ---- D: dose-response in the post era ----
p = d[d.post == 1].copy()
rows = []
for f in ["style_index", "ing_func_1k", "op_additionally_1k", "pct_long_words"]:
    for v in MAIN:
        mv = p.venue == v
        sd = d[(d.venue == v) & (d.era == "pre")][f].std(ddof=0) or 1.0
        mu = d[(d.venue == v) & (d.era == "pre")][f].mean()
        p.loc[mv, "y"] = (p.loc[mv, f] - mu) / sd
    m = smf.ols("y ~ log_llm + log_words + C(venue)", data=p).fit(cov_type="HC3")
    zero = p[p.llm_1k == 0]["y"].mean()
    rows.append({"feature": f, "slope_per_log_unit_sd": m.params["log_llm"], "se": m.bse["log_llm"], "p": m.pvalues["log_llm"],
                 "zero_llm_papers_shift_sd": zero})
pd.DataFrame(rows).to_csv(os.path.join(out, "stats2_dose.csv"), index=False)
summary["dose"] = rows
print("D", pd.DataFrame(rows).round(3).to_string())

# ---- T: non-LLM threshold sensitivity ----
rows = []
for lab, cond in [("all post papers", lambda x: x.llm_1k >= 0), ("< 5 mentions (main analysis)", lambda x: x.llm_topic == 0),
                  ("< 1 per 1k words", lambda x: x.llm_1k < 1), ("zero mentions", lambda x: x.llm_1k == 0)]:
    vals = []
    for v in MAIN:
        a = d[(d.venue == v) & (d.era == "pre")]["style_index"].values
        b = d[(d.venue == v) & (d.era == "post") & cond(d)]["style_index"].values
        if len(b) >= 10:
            vals.append(S.auc(a, b))
    rows.append({"post_papers": lab, "pooled_auc": float(np.mean(vals)), "venues": len(vals),
                 "n": int(((d.era == "post") & cond(d)).sum())})
pd.DataFrame(rows).to_csv(os.path.join(out, "stats2_threshold.csv"), index=False)
summary["threshold"] = rows
print("T", pd.DataFrame(rows).round(3).to_string())

# ---- C: classifier permutation test ----
X = S.zfeat(d, S.STYLE)
y = d.post.values
skf = StratifiedKFold(5, shuffle=True, random_state=0)
clf = LogisticRegression(C=0.3, max_iter=2000)
real = roc_auc_score(y, cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1])
perm = []
for _ in range(200):
    yp = d.groupby("venue")["post"].transform(lambda s: RNG.permutation(s.values)).values
    perm.append(roc_auc_score(yp, cross_val_predict(clf, X, yp, cv=skf, method="predict_proba")[:, 1]))
summary["classifier_perm"] = {"real_auc": real, "perm_mean": float(np.mean(perm)), "perm_max": float(np.max(perm)),
                              "p": (np.sum(np.array(perm) >= real) + 1) / (len(perm) + 1)}
print("C", summary["classifier_perm"])

# ---- L: leave-one-venue-out pooled AUC ----
full, per = pooled_auc(main, "pre", "post", "style_index")
loo = [float(np.mean([x for j, x in enumerate(per) if j != i])) for i in range(len(per))]
summary["lovo_pooled"] = {"full": full, "min": min(loo), "max": max(loo), "per_venue": dict(zip(MAIN, per))}
print("L", summary["lovo_pooled"])
json.dump(summary, open(os.path.join(out, "stats2_summary.json"), "w"), indent=1, default=float)
