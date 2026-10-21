#!/usr/bin/env python3
"""Analyze the LLM-rewrite calibration (see rewrite.py).

Usage: python calib.py <workdir> <outdir> [<figdir>]
For every text group (pre-LLM originals, post-GPT-4 paragraphs, each model x prompt rewrite) we
pool counts over paragraphs and compute rates, with bootstrap 95% CIs over paragraphs. For each
rewrite we report the log ratio vs. its own originals, and compare it with the corpus log ratio
(post-GPT-4 paragraphs vs. pre-LLM paragraphs): sign agreement, Spearman correlation across
features, and the median share of the corpus shift reproduced.
"""
import json, math, os, random, re, sys
import numpy as np
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import WORD, sentences, ing_riders, BOOST, HEDGE, WE, STYLE_LEX, FAMOUS  # noqa: E402
from newfeat import NOMINAL, PASSIVE, syllables  # noqa: E402
from pairs import RX as PAIR_RX  # noqa: E402

wd, out = sys.argv[1], sys.argv[2]
figdir = sys.argv[3] if len(sys.argv) > 3 else None
RW = os.path.join(wd, "rewrite")
MARK = json.load(open(os.path.join(out, "excess_markers.json")))["markers"]
MARK_RX = re.compile(r"\b(" + "|".join(map(re.escape, MARK)) + r")\b", re.I)

METRICS = [  # (key, label, kind) kind: rate per 1k words, or ratio of totals
    ("func", "functional riders /1k", "rate"), ("core", "interpretive riders /1k", "rate"),
    ("addnot", "“Additionally/Notably” openers /1k", "rate"), ("boost", "boosters /1k", "rate"),
    ("hedge", "hedges /1k", "rate"), ("we", "we/our/us /1k", "rate"), ("style", "style words /1k", "rate"),
    ("famous", "famous ChatGPT words /1k", "rate"), ("markers", "split-half marker words /1k", "rate"),
    ("nominal", "nominalizations /1k", "rate"), ("long", "share of 3+ syllable words", "ratio"),
    ("elevated", "elevated share of synonym pairs", "ratio2"), ("passive", "passive sentences per 100", "ratio3"),
    ("sentlen", "mean sentence length (words)", "ratio4"), ("flesch", "Flesch reading ease", "flesch"),
]


def counts(t):
    t = re.sub(r"\s+", " ", t or "").strip()
    w = WORD.findall(t)
    s = sentences(t)
    _, core, func, _, _ = ing_riders(s)
    syl = [syllables(x) for x in w]
    plain = elev = 0
    for _, a, b in PAIR_RX:
        plain += len(a.findall(t))
        elev += len(b.findall(t))
    return {"words": len(w), "sents": max(len(s), 1), "syll": sum(syl), "long": sum(x >= 3 for x in syl),
            "func": func, "core": core, "addnot": sum(bool(re.match(r"(Additionally|Notably)\b", x)) for x in s),
            "boost": len(BOOST.findall(t)), "hedge": len(HEDGE.findall(t)), "we": len(WE.findall(t)),
            "style": len(STYLE_LEX.findall(t)), "famous": len(FAMOUS.findall(t)), "markers": len(MARK_RX.findall(t)),
            "nominal": len(NOMINAL.findall(t)), "passive": sum(bool(PASSIVE.search(x)) for x in s),
            "plain": plain, "elev": elev}


def rates(cs):
    T = {k: sum(c[k] for c in cs) for k in cs[0]}
    W = max(T["words"], 1)
    r = {k: 1000 * T[k] / W for k in ("func", "core", "addnot", "boost", "hedge", "we", "style", "famous", "markers", "nominal")}
    r["long"] = T["long"] / W
    r["elevated"] = T["elev"] / max(T["plain"] + T["elev"], 1)
    r["passive"] = 100 * T["passive"] / T["sents"]
    r["sentlen"] = W / T["sents"]
    r["flesch"] = 206.835 - 1.015 * (W / T["sents"]) - 84.6 * (T["syll"] / W)
    return r


def boot(cs, B=500, seed=0):
    rnd = random.Random(seed)
    res = [rates([rnd.choice(cs) for _ in cs]) for _ in range(B)]
    return {k: (float(np.percentile([x[k] for x in res], 2.5)), float(np.percentile([x[k] for x in res], 97.5))) for k in res[0]}


def lr(a, b, k):
    if k == "flesch":                    # difference, scaled to a log-like unit (points / 10)
        return (b - a) / 10
    eps = 0.02 if k in ("long", "elevated") else 0.05
    return math.log((b + eps) / (a + eps))


paras = json.load(open(os.path.join(RW, "paragraphs.json")))
pre_c = {p["key"]: counts(p["text"]) for p in paras["pre"]}
post_c = [counts(p["text"]) for p in paras["post"]]
pre_all = list(pre_c.values())
R = {"pre-LLM paragraphs (2019–20)": rates(pre_all), "post-GPT-4 paragraphs (2024–26)": rates(post_c)}
CI = {"pre-LLM paragraphs (2019–20)": boot(pre_all), "post-GPT-4 paragraphs (2024–26)": boot(post_c)}
corpus_lr = {k: lr(R["pre-LLM paragraphs (2019–20)"][k], R["post-GPT-4 paragraphs (2024–26)"][k], k) for k, _, _ in METRICS}

conds = []
for fn in sorted(os.listdir(RW)):
    m = re.match(r"out_(\w+)_(\w+)\.json$", fn)
    if not m:
        continue
    rows = [r for r in json.load(open(os.path.join(RW, fn))) if r.get("rewrite")]
    if len(rows) < 20:
        continue
    model, prompt = m.groups()
    rc = [counts(r["rewrite"]) for r in rows]
    oc = [pre_c[r["key"]] for r in rows]
    name = f"{model} · {prompt}"
    R[name], CI[name] = rates(rc), boot(rc)
    orig = rates(oc)
    l = {k: lr(orig[k], R[name][k], k) for k, _, _ in METRICS}
    # Paragraph-level paired test on the main markers (rewrite minus original, per paragraph).
    keys = [k for k, _, _ in METRICS]
    x = np.array([corpus_lr[k] for k in keys])
    y = np.array([l[k] for k in keys])
    sig = [k for k in keys if abs(corpus_lr[k]) > 0.1]       # features with a real corpus shift
    agree = sum(np.sign(corpus_lr[k]) == np.sign(l[k]) for k in sig)
    share = [l[k] / corpus_lr[k] for k in sig if np.sign(l[k]) == np.sign(corpus_lr[k])]
    conds.append({"condition": name, "model": model, "prompt": prompt, "n": len(rows), "lr": l,
                  "spearman": float(st.spearmanr(x, y).correlation), "sign_agree": int(agree), "n_shifted": len(sig),
                  "median_share_of_corpus_shift": float(np.median(share)) if share else 0.0,
                  "orig_words": sum(c["words"] for c in oc), "rewrite_words": sum(c["words"] for c in rc)})

table = []
for k, lab, _ in METRICS:
    row = {"metric": lab, "key": k, "corpus_log_ratio": corpus_lr[k]}
    for g in R:
        row[g] = R[g][k]
        row[g + " lo"], row[g + " hi"] = CI[g][k]
    for c in conds:
        row["lr " + c["condition"]] = c["lr"][k]
    table.append(row)
json.dump({"metrics": METRICS, "rates": R, "ci": CI, "corpus_lr": corpus_lr, "conditions": conds,
           "n_pre": len(pre_all), "n_post": len(post_c)},
          open(os.path.join(out, "rewrite_calibration.json"), "w"), indent=1, default=float)
import csv
with open(os.path.join(out, "rewrite_calibration.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, list(table[0]))
    w.writeheader()
    for r in table:
        w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
print(f"{'metric':38s} {'corpus':>7s} " + " ".join(f"{c['condition'][:16]:>16s}" for c in conds))
for k, lab, _ in METRICS:
    print(f"{lab:38s} {corpus_lr[k]:+7.2f} " + " ".join(f"{c['lr'][k]:+16.2f}" for c in conds))
for c in conds:
    print(f"{c['condition']:18s} n={c['n']:3d} spearman={c['spearman']:+.2f} sign={c['sign_agree']}/{c['n_shifted']} "
          f"share={c['median_share_of_corpus_shift']:.2f} words {c['orig_words']}->{c['rewrite_words']}")

if figdir:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    show = [c for c in conds if c["prompt"] in ("polish", "improve", "grammar", "rewrite", "draft")]
    keys = [k for k, _, _ in METRICS]
    labs = [l for _, l, _ in METRICS]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    yy = np.arange(len(keys))[::-1]
    ax.barh(yy, [corpus_lr[k] for k in keys], color="#d9d9d9", height=0.7, label="real shift, 2019–20 → 2024–26 paragraphs")
    cols = {"polish": "#c0392b", "improve": "#e67e22", "grammar": "#2471a3", "rewrite": "#7d3c98", "draft": "#1e8449"}
    marks = {"haiku": "o", "sonnet": "s", "opus": "D"}
    for c in show:
        ax.scatter([c["lr"][k] for k in keys], yy, s=22, color=cols[c["prompt"]], marker=marks.get(c["model"], "o"),
                   label=c["condition"], zorder=3, alpha=0.85)
    ax.axvline(0, color="#555", lw=0.6)
    ax.set_yticks(yy)
    ax.set_yticklabels(labs)
    ax.set_xlabel("log ratio vs. the original 2019–20 paragraphs (Flesch: difference / 10)")
    ax.legend(fontsize=6.5, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("Does LLM rewriting of 2019–20 paragraphs reproduce the 2024–26 shift?", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig8_rewrite_calibration.png"), dpi=150)
    print("figure written")
