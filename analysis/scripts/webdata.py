#!/usr/bin/env python3
"""Export the numbers the website shows into one JSON file.

Usage: python webdata.py <workdir> <results_dir> <out.json>
Needs <workdir>/all_features.csv (from newfeat.py) and the CSVs in <results_dir>.
"""
import json, math, os, re, sys
import numpy as np
import pandas as pd

wd, R, out = sys.argv[1], sys.argv[2], sys.argv[3]
rd = lambda n: pd.read_csv(os.path.join(R, n))
MAIN = ["MLSys", "OSDI", "NSDI", "VLDB", "NeurIPS", "ICML", "CVPR", "ACL", "ICLR", "AAAI"]
df = pd.read_csv(os.path.join(wd, "all_features.csv")).copy()
df["addnot_1k"] = df["op_additionally_1k"] + df["op_notably_1k"]


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if math.isnan(x) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


D = {"n_papers": int(len(df)), "n_words": int(df["words"].sum())}

# ---- yearly series for ACL and ICML ----
SER = {"ing_func_1k": "Functional rider (“, enabling…”) per 1k words",
       "ing_interp_core_1k": "Interpretive rider (“, highlighting…”) per 1k words",
       "op_additionally_1k": "Sentences opening “Additionally,” per 1k words",
       "op_notably_1k": "Sentences opening “Notably,” per 1k words",
       "boost_1k": "Boosters (significantly, substantially…) per 1k words",
       "hedge_1k": "Hedges (may, might, likely…) per 1k words",
       "we_1k": "we / our / us per 1k words",
       "style_lex_1k": "Style words (crucial, enhance, seamless…) per 1k words",
       "pct_long_words": "Share of words with 3+ syllables",
       "flesch": "Flesch reading ease (higher = easier)",
       "passive_per_100s": "Passive sentences per 100",
       "contrib_introduce_share": "“introduce” as share of contribution verbs",
       "emdash_1k": "Em-dashes per 1k words",
       "sent_len": "Mean sentence length (words)",
       "llm_1k": "LLM mentions per 1k words"}
VS = ["ACL", "ICML", "CVPR", "NeurIPS", "VLDB", "MLSys", "OSDI", "NSDI", "ICLR", "AAAI"]
YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
s = df[df.venue.isin(VS) & df.year.isin(YEARS)].copy()
s.loc[(s.venue == "MLSys") & (s.year == 2019), "year"] = 2020   # MLSys baseline pools 2019 and 2020


def pack(g):
    """g: DataFrame indexed by year with one column per venue -> {years, venues{v:[...]}, pooled[...]}"""
    out = {"years": YEARS, "venues": {}, "pooled": []}
    for v in VS:
        out["venues"][v] = [None if (v not in g or y not in g.index or pd.isna(g.loc[y, v])) else float(g.loc[y, v]) for y in YEARS]
    for i, y in enumerate(YEARS):
        vals = [out["venues"][v][i] for v in VS if out["venues"][v][i] is not None]
        out["pooled"].append(float(np.mean(vals)) if len(vals) >= 5 else None)
    return out


series = {}
for f, lab in SER.items():
    g = s.groupby(["year", "venue"])[f].mean().unstack(1)
    series[f] = {"label": lab, **pack(g)}
cls = rd("stats_D_by_year.csv")
c = cls.copy()
c.loc[(c.venue == "MLSys") & (c.year == 2019), "year"] = 2020
g = c.groupby(["year", "venue"]).apply(lambda x: np.average(x.share_gt_half, weights=x.n)).unstack(1)
series["classifier"] = {"label": "Papers a style-only classifier calls post-GPT-4 (share)", **pack(g)}
ex = rd("excess.csv").drop_duplicates("corpus")
ex["year"] = ex.corpus.str[-4:].astype(int)
ex["v"] = ex.corpus.str.extract(r"^([a-z]+)")[0].map({"acl": "ACL", "icml": "ICML", "cvpr": "CVPR", "neurips": "NeurIPS", "vldb": "VLDB",
                                                      "mlsys": "MLSys", "osdi": "OSDI", "nsdi": "NSDI", "iclr": "ICLR", "aaai": "AAAI"})
ex = ex[ex.v.notna()].copy()
ex.loc[(ex.v == "MLSys") & (ex.year == 2019), "year"] = 2020
g = ex.groupby(["year", "v"]).apply(lambda x: np.average(x.share_3plus_markers, weights=x.n)).unstack(1)
series["markers"] = {"label": "Papers using 3+ marker words (underscore, showcase, overlook…)", **pack(g)}
D["series"] = series

# ---- venues ----
pv = rd("per_venue.csv")
npv = rd("new_per_venue.csv")
si = rd("style_index.csv")
mix = rd("stats_E_mixture.csv")
ts = rd("topic_share.csv").set_index("corpus")
VF = [("ing_func_1k", "Functional riders /1k", pv), ("ing_interp_core_1k", "Interpretive riders /1k", pv),
      ("op_additionally_1k", "“Additionally,” /1k", pv), ("op_notably_1k", "“Notably,” /1k", pv),
      ("boost_1k", "Boosters /1k", pv), ("hedge_1k", "Hedges /1k", pv), ("we_1k", "we/our/us /1k", pv),
      ("style_lex_1k", "Style words /1k", pv), ("pct_long_words", "3+ syllable words", npv),
      ("flesch", "Flesch reading ease", npv), ("passive_per_100s", "Passive per 100 sentences", npv),
      ("extensive_exp", "Says “extensive experiments”", npv), ("sent_len", "Sentence length", pv),
      ("emdash_1k", "Em-dashes /1k", pv), ("has_limitations", "Has a Limitations section", pv),
      ("n_refs", "References", pv), ("pages", "PDF pages", pv)]
venues = {}
for v in MAIN:
    rows = []
    for f, lab, src in VF:
        if src is pv:
            r = pv[(pv.venue == v) & (pv.feature == f)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({"f": f, "label": lab, "c2016": r.ctrl_mean if r.n_ctrl else None, "pre": r.pre_mean,
                         "post": r.post_mean, "auc": r.auc})
        else:
            r = npv[(npv.venue == v) & (npv.feature == f)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({"f": f, "label": lab, "c2016": r.c2016, "pre": r.pre, "post": r.post, "auc": r.auc})
    sv = si[si.venue == v].sort_values("year")
    post_c = [c for c in sv.corpus if int(c[-4:]) >= 2024]
    last = sv[sv.corpus == post_c[-1]].iloc[0]
    mx = mix[(mix.venue == v) & mix.corpora.str.contains(r"\+|2024|2025") ].iloc[0] if v not in ("ACL", "ICML") else \
        mix[(mix.venue == v) & (mix.corpora == ("acl2025" if v == "ACL" else "icml2025"))].iloc[0]
    pre_c = [c for c in sv.corpus if 2019 <= int(c[-4:]) <= 2020]
    s2 = json.load(open(os.path.join(R, "stats2_summary.json")))
    venues[v] = {"features": rows, "tail_share": last.share_above_pre_p95, "tail_corpus": last.corpus,
                 "index_auc": s2["lovo_pooled"]["per_venue"][v],
                 "mixture": mx.share_shifted, "mixture_lo": mx.ci_lo, "mixture_hi": mx.ci_hi,
                 "llm_pre": float(np.mean([ts.loc[c, "llm_topic_share"] for c in pre_c])),
                 "llm_post": float(np.mean([ts.loc[c, "llm_topic_share"] for c in post_c])),
                 "n_pre": int(pv[(pv.venue == v)].iloc[0].n_pre), "n_post": int(pv[(pv.venue == v)].iloc[0].n_post)}
D["venues"] = venues
D["tail_by_corpus"] = si[["venue", "corpus", "year", "share_above_pre_p95", "p10", "p50", "p90"]].to_dict("records")

# ---- synonym ladder ----
sp = rd("synonym_pairs_summary.csv")
D["pairs"] = sp[["pair", "2016", "pre", "post", "post_nonllm"]].to_dict("records")

# ---- vocabulary ----
vc = rd("vocab.csv")
maxv = vc.venues.max()
common = vc[(vc.post_df >= 0.06) & (vc.venues >= maxv)]
topic_rx = re.compile(r"large language|llm|gpt|llama|qwen|prompt|token|diffusion|transformer|openai|foundation|vllm|reasoning|agent|"
                      r"language model|models llms|sota|rtx|touvron|multimodal|kv|instruction|fine-tun|deepseek|claude|gemini|mllm|lora|rlhf")
style = common[(common.mean_lr_nonllm > 1.3) & (common.trend_lr < 0.45) & (common.venues_up >= maxv - 1)
               & ~common.term.str.contains(topic_rx)]
style = style.sort_values("mean_lr_nonllm", ascending=False)
style = style[~style.term.str.match(r"^(the|of|and|to|in|a|for|its|with|on|by) ")].head(28)
topic = common[common.term.str.contains(topic_rx) & (common.mean_lr_nonllm < 0.65 * common.mean_lr)].sort_values(
    "mean_lr", ascending=False)
topic = topic[~topic.term.str.contains(" ")].head(12)
fall = vc[(vc.pre_df >= 0.08) & (vc.venues >= maxv)].sort_values("mean_lr")
FALL_TOPIC = re.compile(r"tensorflow|rnn|recurrent|facebook|http|creativecommons|visit|lstm|mini-batch|fully-connected|"
                        r"network|machine translation|classifier|word|data set|cnn|convolutional|caffe|theano|lua|by-nc|commons|attribution|noderivatives|license")
fall_style = fall[~fall.term.str.contains(FALL_TOPIC)].head(24)
D["vocab"] = {"style": style[["term", "pre_df", "post_df", "mean_lr", "mean_lr_nonllm", "trend_lr"]].to_dict("records"),
              "topic": topic[["term", "pre_df", "post_df", "mean_lr", "mean_lr_nonllm"]].to_dict("records"),
              "fall": fall_style[["term", "pre_df", "post_df", "mean_lr", "mean_lr_nonllm", "trend_lr"]].to_dict("records")}

# ---- statistics ----
D["did"] = rd("stats_B_did.csv")[["feature", "naive_jump_sd", "excess_jump_sd", "excess_se", "excess_p"]].to_dict("records")
D["did_new"] = rd("new_pooled.csv")[["feature", "pooled_auc", "excess_jump_sd", "excess_se", "best_tau", "delta_bic_vs_linear"]].to_dict("records")
D["changepoint"] = rd("stats_C_changepoint.csv").to_dict("records")
D["classifier"] = rd("stats_D_classifier.csv").to_dict("records")
D["mixture"] = rd("stats_E_mixture.csv").to_dict("records")
D["coadoption"] = rd("stats_F_coadoption.csv").to_dict("records")
D["variance"] = rd("stats_G_variance.csv").to_dict("records")
D["sections"] = rd("stats_H_sections.csv").to_dict("records")
pool = rd("stats_A_auc_ci_fdr.csv")
pv_tests = pool[pool.venue != "POOLED"]
sig = pv_tests[pv_tests.q_bh < 0.05]
D["fdr"] = {"tests": int(len(pv_tests)), "sig": int(len(sig)), "increases": int((sig.auc > 0.5).sum()),
            "all_venues": int((sig.groupby("feature").size() == len(MAIN)).sum())}
D["pooled_auc"] = pool[pool.venue == "POOLED"][["feature", "auc", "ci_lo", "ci_hi", "q_bh"]].to_dict("records")

# ---- meter: the same index the page computes from pasted text ----
MF = [("ing_func_1k", 1), ("ing_interp_core_1k", 1), ("boost_1k", 1), ("hedge_1k", -1), ("addnot_1k", 1),
      ("style_lex_1k", 1), ("we_1k", -1), ("pct_long_words", 1)]
main = df[df.venue.isin(MAIN)]
pre, post = main[main.era == "pre"], main[main.era == "post"]
stats = {f: (pre[f].mean(), pre[f].std(ddof=0) or 1.0) for f, _ in MF}


def index(d):
    z = [sign * ((d[f] - stats[f][0]) / stats[f][1]).clip(-5, 5) for f, sign in MF]
    return sum(z) / len(z)


ip, iq = index(pre), index(post)
D["meter"] = {"features": [{"f": f, "sign": s, "mean": stats[f][0], "sd": stats[f][1], "post_mean": float(post[f].mean()),
                            "pre_median": float(pre[f].median()), "post_median": float(post[f].median())} for f, s in MF],
              "pre_q": [float(np.percentile(ip, q)) for q in range(0, 101, 5)],
              "post_q": [float(np.percentile(iq, q)) for q in range(0, 101, 5)]}
json.dump(clean(D), open(out, "w"))
print("wrote", out, os.path.getsize(out), "bytes")
