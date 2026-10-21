#!/usr/bin/env python3
"""Statistical inference on top of analyze.py's per-paper features.

Usage: python stats.py <workdir> <outdir> [A..H P]   (needs numpy, scipy, pandas, statsmodels, scikit-learn)
Reads <workdir>/features.jsonl and <workdir>/text/; writes stats_*.csv and stats_summary.md to <outdir>.

  A  per-venue AUC with bootstrap 95% CI, Mann-Whitney p, Benjamini-Hochberg FDR over all tests;
     pooled AUC with a venue-stratified bootstrap CI
  B  trend-adjusted difference-in-differences: does the post-AI value exceed the linear
     extrapolation of the 2016->2020(->2023) trend? OLS with venue fixed effects, HC3 SEs
  C  changepoint: ACL+ICML 2016-2025, step model at each candidate year vs. linear trend (BIC),
     bootstrap distribution of the best year
  D  style-only classifier (pre vs post): 5-fold CV AUC, leave-one-venue-out AUC, and predicted
     post-likeness for held-out 2016 and 2021-2024 papers
  E  mixture: post = (1-a) * f_pre + a * N(mu, s); share a of papers that left the pre-AI distribution
  F  co-adoption: mean pairwise Spearman correlation of the markers within era (venue-demeaned)
  G  variance decomposition of the style index: era vs venue vs LLM topic (type-II ANOVA)
  H  where in the paper: marker rates per section (abstract, intro, related work, conclusion)
"""
import json, os, re, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import VENUES, SERIES  # noqa: E402
import features as F  # noqa: E402

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(0)
MAIN = [v for v in VENUES if not v.endswith("*")]
# Style features only: no topic counts, no structure, no ICML-artifact paragraph length.
STYLE = ["ing_func_1k", "ing_interp_core_1k", "ing_interp_1k", "ing_other_1k", "boost_1k", "hedge_1k", "we_1k",
         "op_additionally_1k", "op_notably_1k", "op_specifically_1k", "op_furthermore_1k", "op_however_1k",
         "op_moreover_1k", "op_while_1k", "op_thus_1k", "op_in_particular_1k", "this_closer_1k", "emdash_1k",
         "rather_than_1k", "triad_1k", "not_only_1k", "semicolon_1k", "colon_1k", "question_1k", "sent_len",
         "sent_cv", "runin_share", "style_lex_1k", "famous_ai_words_1k", "mattr", "novel_1k",
         "first_claim_1k", "quant_1k"]
INDEX9 = ["ing_interp_core_1k", "ing_func_1k", "boost_1k", "op_additionally_1k", "op_notably_1k",
          "emdash_1k", "this_closer_1k", "style_lex_1k", "famous_ai_words_1k"]
KEY = ["ing_func_1k", "ing_interp_core_1k", "boost_1k", "hedge_1k", "we_1k", "op_additionally_1k",
       "op_notably_1k", "style_lex_1k", "this_closer_1k", "emdash_1k", "mattr", "sent_len", "sent_cv",
       "triad_1k", "not_only_1k", "style_index"]


def year(c):
    return int(re.search(r"\d{4}", c).group())


def load(wd):
    rows = [json.loads(l) for l in open(os.path.join(wd, "features.jsonl"))]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "fw"} for r in rows])
    df["year"] = df["corpus"].map(year)
    venue, era = {}, {}
    for v in MAIN:
        ctrl, pre, post = VENUES[v]
        for c in ctrl:
            venue[c], era[c] = v, "2016"
        for c in pre:
            venue[c], era[c] = v, "pre"
        for c in post:
            venue[c], era[c] = v, "post"
    for v, cs in SERIES.items():
        for c in cs[1:-1]:
            venue[c], era[c] = v, "series"
        for y in (2017, 2018, 2019):        # pre-LLM trend years (mlsys2019 is part of the baseline)
            ct = re.sub(r"\d{4}$", str(y), cs[0])
            if ct != "mlsys2019":
                venue[ct], era[ct] = v, "trend"
    df["venue"] = df["corpus"].map(venue)
    df["era"] = df["corpus"].map(era)
    df = df[df["venue"].notna()].copy()
    # Prespecified style index (as in analyze.py): z vs the venue's pre-AI mean, pre+post SD.
    df["style_index"] = 0.0
    for v in MAIN:
        m = df.venue == v
        pre, both = df[m & (df.era == "pre")], df[m & df.era.isin(["pre", "post"])]
        z = [(df.loc[m, f] - pre[f].mean()) / max(both[f].std(ddof=0), 1e-3) for f in INDEX9]
        df.loc[m, "style_index"] = sum(z) / len(z)
    return df


def auc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    r = stats.rankdata(np.concatenate([a, b]))
    return (r[len(a):].sum() - len(b) * (len(b) + 1) / 2) / (len(a) * len(b))


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


# ---------------- A ----------------
def part_a(df, B=1000):
    rows = []
    feats = STYLE + ["style_index", "pages", "n_refs", "has_limitations", "contrib_bullets", "words"]
    for f in feats:
        per = {}
        for v in MAIN:
            a = df[(df.venue == v) & (df.era == "pre")][f].values
            b = df[(df.venue == v) & (df.era == "post")][f].values
            per[v] = (a, b)
            boots = [auc(RNG.choice(a, len(a)), RNG.choice(b, len(b))) for _ in range(B // 4)]
            p = stats.mannwhitneyu(b, a, alternative="two-sided").pvalue
            rows.append({"feature": f, "venue": v, "auc": auc(a, b), "ci_lo": np.percentile(boots, 2.5),
                         "ci_hi": np.percentile(boots, 97.5), "p": p, "n_pre": len(a), "n_post": len(b)})
        pooled = np.mean([auc(a, b) for a, b in per.values()])
        pb = [np.mean([auc(RNG.choice(a, len(a)), RNG.choice(b, len(b))) for a, b in per.values()])
              for _ in range(B // 4)]
        # Stratified permutation test of the pooled AUC (labels shuffled within venue).
        perm = []
        for _ in range(200):
            vals = []
            for a, b in per.values():
                x = RNG.permutation(np.concatenate([a, b]))
                vals.append(auc(x[:len(a)], x[len(a):]))
            perm.append(np.mean(vals))
        pp = (np.sum(np.abs(np.array(perm) - 0.5) >= abs(pooled - 0.5)) + 1) / (len(perm) + 1)
        rows.append({"feature": f, "venue": "POOLED", "auc": pooled, "ci_lo": np.percentile(pb, 2.5),
                     "ci_hi": np.percentile(pb, 97.5), "p": pp, "n_pre": None, "n_post": None})
    out = pd.DataFrame(rows)
    m = out.venue != "POOLED"
    out.loc[m, "q_bh"] = bh(out.loc[m, "p"])
    out.loc[~m, "q_bh"] = bh(out.loc[~m, "p"])
    return out


# ---------------- B ----------------
def part_b(df):
    """Trend-adjusted DiD on all eight venues. y standardized by the venue's pre-LLM SD.
    Model: y ~ C(venue) + C(venue):t + post, t = year - 2020, fitted on years <= 2022 (pre-ChatGPT:
    2016, 2018, 2019-20, 2021, 2022) plus the post years >= 2024; 2023 is excluded as mixed.
    Standard errors are clustered by venue-year, since papers from the same venue-year share shocks."""
    rows = []
    for f in KEY:
        d = df[df.venue.isin(MAIN) & ((df.year <= 2022) | (df.year >= 2024))].copy()
        d = d[d[f].notna()]
        for v in MAIN:
            m = d.venue == v
            pre = d[m & (d.era == "pre")][f]
            sd = pre.std(ddof=0) or 1.0
            d.loc[m, "y"] = (d.loc[m, f] - pre.mean()) / sd
        d["t"] = d["year"] - 2020
        d["post"] = (d["year"] >= 2024).astype(int)
        d["cell"] = d["venue"] + d["year"].astype(str)
        g = d["cell"].astype("category").cat.codes
        res = smf.ols("y ~ C(venue) + C(venue):t + post", data=d).fit(cov_type="cluster", cov_kwds={"groups": g})
        dn = d[d.era.isin(["pre", "post"])]
        naive = smf.ols("y ~ C(venue) + post", data=dn).fit(cov_type="cluster",
                                                           cov_kwds={"groups": dn["cell"].astype("category").cat.codes})
        tr = np.mean([res.params.get(f"C(venue)[{v}]:t", 0.0) for v in MAIN])
        rows.append({"feature": f, "naive_jump_sd": naive.params["post"], "naive_se": naive.bse["post"],
                     "excess_jump_sd": res.params["post"], "excess_se": res.bse["post"],
                     "excess_p": res.pvalues["post"], "mean_pre_trend_sd_per_yr": tr, "n": len(d),
                     "clusters": int(d["cell"].nunique())})
    return pd.DataFrame(rows)


# ---------------- C ----------------
def fit_bic(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(((y - X @ beta) ** 2).sum())
    n, k = X.shape
    return n * np.log(rss / n) + k * np.log(n), beta


def changepoint(d, f):
    """Step-vs-trend on venue-year cell means (the conservative unit: one point per venue-year),
    venue intercepts, common slope; tau = first affected year."""
    c = d.groupby(["venue", "year"])[f].mean().reset_index()
    y = c[f].values.astype(float)
    t = (c["year"] - 2020).values.astype(float)
    vs = sorted(c["venue"].unique())
    dums = [(c["venue"] == v).values.astype(float) for v in vs[1:]]
    base = np.column_stack([np.ones_like(t), *dums, t])
    bic_lin, _ = fit_bic(y, base)
    res = {}
    for tau in range(2021, 2026):
        X = np.column_stack([base, (c["year"].values >= tau).astype(float)])
        b, beta = fit_bic(y, X)
        res[tau] = (b, beta[-1])
    best = min(res, key=lambda k: res[k][0])
    return bic_lin, res, best


def part_c(df, B=200, venues=None):
    """Bootstrap resamples papers within each venue-year cell, then refits on cell means."""
    venues = venues or list(SERIES)
    d = df[df.venue.isin(venues) & df.era.isin(["2016", "trend", "pre", "series", "post"]) & (df.year <= 2025)].copy()
    rows = []
    for f in KEY:
        dd = d[d[f].notna()]
        bic_lin, res, best = changepoint(dd, f)
        boots = []
        groups = [g for _, g in dd.groupby(["venue", "year"])]
        for _ in range(B):
            s = pd.concat([g.sample(len(g), replace=True, random_state=int(RNG.integers(1e9))) for g in groups])
            boots.append(changepoint(s, f)[2])
        cnt = pd.Series(boots).value_counts(normalize=True)
        cells = dd.groupby(["venue", "year"])[f].mean()
        sd = cells[cells.index.get_level_values(1) <= 2020].std(ddof=0) or 1.0
        rows.append({"feature": f, "best_tau": best, "delta_bic_vs_linear": bic_lin - res[best][0],
                     "step_size_cell_sd": res[best][1] / sd, "cells": int(len(cells)),
                     **{f"bic_tau{k}": res[k][0] - res[best][0] for k in res},
                     **{f"boot_p{k}": cnt.get(k, 0.0) for k in range(2021, 2026)}})
    return pd.DataFrame(rows)


def part_c_per_venue(df, feats=("style_index", "ing_func_1k", "ing_interp_core_1k", "op_notably_1k", "pct_long_words")):
    """Best break year per venue (no bootstrap), for the per-field table."""
    rows = []
    for v in SERIES:
        d = df[(df.venue == v) & df.era.isin(["2016", "trend", "pre", "series", "post"]) & (df.year <= 2025)]
        for f in feats:
            if f not in d:
                continue
            bic_lin, res, best = changepoint(d, f)
            rows.append({"venue": v, "feature": f, "best_tau": best, "delta_bic_vs_linear": bic_lin - res[best][0],
                         "years": ",".join(str(y) for y in sorted(d.year.unique()))})
    return pd.DataFrame(rows)


# ---------------- D ----------------
def zfeat(df, feats):
    X = df[feats].copy().astype(float)
    for v in df.venue.unique():
        m = df.venue == v
        pre = df[m & (df.era == "pre")]
        for f in feats:
            X.loc[m, f] = (X.loc[m, f] - pre[f].mean()) / (pre[f].std(ddof=0) or 1.0)
    return X.clip(-8, 8)


def part_d(df):
    feats = [f for f in STYLE if f not in ("famous_ai_words_1k", "style_lex_1k")]  # syntax + stance only
    sets = {"style (33 features)": STYLE, "syntax+stance, no word lists (31)": feats}
    out, preds = [], None
    for name, fs in sets.items():
        X = zfeat(df, fs)
        tr = df.era.isin(["pre", "post"])
        y = (df.era == "post").astype(int)
        clf = LogisticRegression(C=0.3, max_iter=2000)
        cv = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X[tr], y[tr]):
            clf.fit(X[tr].iloc[a], y[tr].iloc[a])
            cv.append(roc_auc_score(y[tr].iloc[b], clf.predict_proba(X[tr].iloc[b])[:, 1]))
        lovo = {}
        for v in MAIN:
            m_tr = tr & (df.venue != v)
            m_te = tr & (df.venue == v)
            clf.fit(X[m_tr], y[m_tr])
            lovo[v] = roc_auc_score(y[m_te], clf.predict_proba(X[m_te])[:, 1])
        clf.fit(X[tr], y[tr])
        row = {"model": name, "cv_auc": np.mean(cv), "cv_auc_sd": np.std(cv),
               **{f"lovo_{v}": a for v, a in lovo.items()}, "lovo_mean": np.mean(list(lovo.values()))}
        out.append(row)
        if preds is None:
            coef = pd.Series(clf.coef_[0], index=fs).sort_values()
            p = clf.predict_proba(X)[:, 1]
            preds = df[["corpus", "venue", "era", "year", "llm_topic"]].assign(p_post=p)
            coefs = coef
    held = preds[preds.era.notna()]
    by_year = held.groupby(["venue", "year", "era"]).agg(n=("p_post", "size"), mean_p=("p_post", "mean"),
                                                         share_gt_half=("p_post", lambda x: (x > 0.5).mean())).reset_index()
    return pd.DataFrame(out), coefs.rename("coef").reset_index().rename(columns={"index": "feature"}), by_year


# ---------------- E ----------------
def mixture(pre, post, iters=300):
    """EM for post ~ (1-a) * KDE(pre) + a * N(mu, s). Returns a, mu, s."""
    kde = stats.gaussian_kde(pre)
    f0 = np.maximum(kde(post), 1e-12)
    a, mu, s = 0.5, np.mean(post) + np.std(post), np.std(post)
    for _ in range(iters):
        f1 = stats.norm.pdf(post, mu, s)
        r = a * f1 / (a * f1 + (1 - a) * f0)
        a = r.mean()
        mu = (r * post).sum() / max(r.sum(), 1e-9)
        s = max(np.sqrt((r * (post - mu) ** 2).sum() / max(r.sum(), 1e-9)), 0.05)
    return a, mu, s


def part_e(df, B=200):
    rows = []
    groups = [(v, VENUES[v][2]) for v in MAIN] + [(v, [c]) for v, cs in SERIES.items() for c in cs[1:]]
    for v, cs in groups:
        pre = df[(df.venue == v) & (df.era == "pre")]["style_index"].values
        post = df[df.corpus.isin(cs) & (df.venue == v)]["style_index"].values
        if len(post) < 20:
            continue
        a, mu, s = mixture(pre, post)
        boots = [mixture(RNG.choice(pre, len(pre)), RNG.choice(post, len(post)), 150)[0] for _ in range(B // 4)]
        rows.append({"venue": v, "corpora": "+".join(cs), "n_post": len(post), "share_shifted": a,
                     "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
                     "shifted_mean": mu, "shifted_sd": s})
    return pd.DataFrame(rows)


# ---------------- F ----------------
def part_f(df, B=300):
    feats = INDEX9 + ["hedge_1k", "we_1k"]
    out = []
    for era in ["2016", "pre", "post"]:
        d = df[df.era == era]
        if era == "2016":
            vs = [v for v in MAIN if VENUES[v][0]]
        else:
            vs = MAIN
        d = d[d.venue.isin(vs)]
        # Venue-demean ranks so venue differences do not create correlation.
        R = d[feats].rank(pct=True)
        R = R - R.groupby(d.venue).transform("mean")
        c = R.corr().values  # Pearson on venue-demeaned percentile ranks (same as the bootstrap)
        off = c[np.triu_indices(len(feats), 1)]
        boots = []
        idx = np.arange(len(R))
        for _ in range(B // 3):
            s = R.iloc[RNG.choice(idx, len(idx))]
            cc = s.corr().values
            boots.append(cc[np.triu_indices(len(feats), 1)].mean())
        ev = np.linalg.eigvalsh(np.corrcoef(R.values.T))[::-1]
        out.append({"era": era, "n": len(d), "mean_pairwise_rho": off.mean(), "ci_lo": np.percentile(boots, 2.5),
                    "ci_hi": np.percentile(boots, 97.5), "pc1_share": ev[0] / ev.sum()})
    return pd.DataFrame(out)


# ---------------- G ----------------
def part_g(df):
    rows = []
    d = df[df.era.isin(["pre", "post"])].copy()
    import statsmodels.api as sm
    # Raw features only: the style index is centred per venue, which removes venue variance by construction.
    for f in ["ing_func_1k", "ing_interp_core_1k", "op_additionally_1k", "op_notably_1k", "boost_1k", "style_lex_1k",
              "we_1k", "hedge_1k", "sent_len", "mattr"]:
        res = smf.ols(f"{f} ~ C(era) + C(venue) + C(era):C(venue) + llm_topic", data=d).fit()
        an = sm.stats.anova_lm(res, typ=2)
        tot = an["sum_sq"].sum()
        rows.append({"feature": f, **{k: an.loc[k, "sum_sq"] / tot for k in an.index}})
    return pd.DataFrame(rows)


# ---------------- H ----------------
def rates(text):
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    words = len(F.WORD.findall(text))
    if words < 60:
        return None
    sents = F.sentences(text)
    i_all, core, func, _, _ = F.ing_riders(sents)
    k = 1000 / words
    return {"words": words, "func": func * k, "core": core * k, "boost": len(F.BOOST.findall(text)) * k,
            "hedge": len(F.HEDGE.findall(text)) * k, "lex": len(F.STYLE_LEX.findall(text)) * k,
            "addnot": sum(bool(re.match(r"(Additionally|Notably)\b", s)) for s in sents) * k,
            "we": len(F.WE.findall(text)) * k}


def part_h(wd, df):
    keep = {(r.corpus, r.id): (r.venue, r.era) for r in df.itertuples() if r.era in ("pre", "post")}
    rows = []
    for c in sorted({k[0] for k in keep}):
        for fn in os.listdir(os.path.join(wd, "text", c)):
            rec = json.load(open(os.path.join(wd, "text", c, fn)))
            if (c, rec["id"]) not in keep:
                continue
            v, era = keep[(c, rec["id"])]
            secs = {"abstract": rec.get("abstract", "")}
            for h, _, t in rec["segments"]:
                for name, rx in (("introduction", r"^introduction"), ("related work", r"related"),
                                 ("conclusion", r"conclusion|summary"), ("limitations", r"limitation")):
                    if re.search(rx, h) and t:
                        secs[name] = secs.get(name, "") + "\n" + t
            secs["whole body"] = "\n".join(rec["paras"])
            for s, t in secs.items():
                r = rates(t)
                if r:
                    rows.append({"venue": v, "era": era, "section": s, **r})
    d = pd.DataFrame(rows)
    agg = d.groupby(["section", "era"])[["func", "core", "boost", "hedge", "lex", "addnot", "we"]].mean().unstack("era")
    out = []
    for s in agg.index:
        r = {"section": s, "n_pre": int(((d.section == s) & (d.era == "pre")).sum()),
             "n_post": int(((d.section == s) & (d.era == "post")).sum())}
        for m in ["func", "core", "boost", "hedge", "lex", "addnot", "we"]:
            r[f"{m}_pre"], r[f"{m}_post"] = agg.loc[s, (m, "pre")], agg.loc[s, (m, "post")]
            a = d[(d.section == s) & (d.era == "pre")][m]
            b = d[(d.section == s) & (d.era == "post")][m]
            r[f"{m}_auc"] = auc(a.values, b.values)
        out.append(r)
    return pd.DataFrame(out)


def main():
    wd, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    df = load(wd)
    parts = sys.argv[3:] or list("ABCDEFGH")
    if "A" in parts:
        a = part_a(df)
        a.to_csv(os.path.join(out, "stats_A_auc_ci_fdr.csv"), index=False)
        print("A done")
    if "B" in parts:
        part_b(df).to_csv(os.path.join(out, "stats_B_did.csv"), index=False)
        print("B done")
    if "C" in parts:
        part_c(df).to_csv(os.path.join(out, "stats_C_changepoint.csv"), index=False)
        part_c_per_venue(df).to_csv(os.path.join(out, "stats_C_per_venue.csv"), index=False)
        print("C done")
    if "D" in parts:
        m, coefs, by_year = part_d(df)
        m.to_csv(os.path.join(out, "stats_D_classifier.csv"), index=False)
        coefs.to_csv(os.path.join(out, "stats_D_coefs.csv"), index=False)
        by_year.to_csv(os.path.join(out, "stats_D_by_year.csv"), index=False)
        print("D done")
    if "E" in parts:
        part_e(df).to_csv(os.path.join(out, "stats_E_mixture.csv"), index=False)
        print("E done")
    if "F" in parts:
        part_f(df).to_csv(os.path.join(out, "stats_F_coadoption.csv"), index=False)
        print("F done")
    if "G" in parts:
        part_g(df).to_csv(os.path.join(out, "stats_G_variance.csv"), index=False)
        print("G done")
    if "H" in parts:
        part_h(wd, df).to_csv(os.path.join(out, "stats_H_sections.csv"), index=False)
        print("H done")
    if "P" in parts:
        plot(out, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"))
        print("plot done")



def plot(out, figdir):
    """fig7: changepoint BIC, held-out classifier by year, tail share by year, co-adoption."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    C = pd.read_csv(os.path.join(out, "stats_C_changepoint.csv")).set_index("feature")
    Y = pd.read_csv(os.path.join(out, "stats_D_by_year.csv"))
    E = pd.read_csv(os.path.join(out, "stats_E_mixture.csv"))
    Fd = pd.read_csv(os.path.join(out, "stats_F_coadoption.csv"))
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    taus = list(range(2021, 2026))
    for f, lab in [("style_index", "style index"), ("ing_interp_core_1k", "interpretive rider"),
                   ("ing_func_1k", "functional rider"), ("op_notably_1k", "'Notably'"), ("emdash_1k", "em-dash"),
                   ("sent_len", "sentence length")]:
        ax[0].plot(taus, [C.loc[f, f"bic_tau{t}"] for t in taus], marker="o", ms=3, label=lab)
    ax[0].set_yscale("symlog")
    ax[0].set_xticks(taus)
    ax[0].set_ylabel("ΔBIC vs best break year (0 = best)")
    ax[0].set_title("(a) Which year does the step start? (10 venues)", fontsize=8)
    ax[0].legend(fontsize=6, frameon=False)
    for v, col in (("ACL", "#17becf"), ("ICML", "#8c564b")):
        d = Y[Y.venue == v].sort_values("year")
        ax[1].plot(d.year, d.share_gt_half * 100, marker="o", color=col, label=v)
    for v in ("MLSys", "OSDI", "NSDI", "VLDB", "NeurIPS", "CVPR"):
        d = Y[Y.venue == v].sort_values("year")
        ax[1].plot(d.year, d.share_gt_half * 100, marker=".", ls=":", lw=0.8, label=v)
    ax[1].axvspan(2022.9, 2026.2, color="#f5b7b1", alpha=0.25, lw=0)
    ax[1].set_ylabel("% papers classified post-GPT-4 (p > 0.5)")
    ax[1].set_title("(b) Style classifier; 2016, 2018, 2021-24 held out", fontsize=8)
    ax[1].legend(fontsize=5.5, frameon=False, ncol=2)
    SI = pd.read_csv(os.path.join(out, "style_index.csv"))
    SI = SI[~SI.venue.str.endswith("*")].copy()
    SI["y"] = SI.corpus.str[-4:].astype(int)
    SI.loc[SI.corpus == "mlsys2019", "y"] = 2020
    g = SI[SI.y <= 2025].groupby("y").share_above_pre_p95.agg(["mean", "count"]).reset_index()
    cols = ["#2471a3" if y <= 2022 else ("#999999" if y == 2023 else "#c0392b") for y in g.y]
    ax[2].bar(g.y.astype(str), g["mean"] * 100, color=cols)
    ax[2].axhline(5, color="#555", lw=0.7, ls="--")
    ax[2].set_ylabel("% papers above the venue's 2019–20 top 5%")
    ax[2].set_title("(c) Placebo years vs. post-GPT-4 (5% = no change)", fontsize=8)
    ax[3].bar(Fd.era.astype(str), Fd.mean_pairwise_rho, yerr=[Fd.mean_pairwise_rho - Fd.ci_lo, Fd.ci_hi - Fd.mean_pairwise_rho],
              color=["#999", "#2471a3", "#c0392b"])
    ax[3].set_ylabel("mean pairwise correlation of 11 markers")
    ax[3].set_title("(d) Markers co-occur after GPT-4", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig7_statistics.png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
