#!/usr/bin/env python3
"""Figures for the paper, from the CSVs written by analyze.py. Needs matplotlib + numpy.

Usage: python figures.py <results_dir> <fig_dir>
"""
import csv, math, os, re, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R, F = sys.argv[1], sys.argv[2]
os.makedirs(F, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tables import LABEL  # noqa: E402  (tables.py only reads argv[1] when run as a script)

rd = lambda n: list(csv.DictReader(open(os.path.join(R, n))))
fl = lambda x: float(x) if x not in ("", "nan", None) else float("nan")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
COL = {"AI-era break": "#c0392b", "pre-AI trend": "#2471a3", "topic": "#7d3c98",
       "gradual (mostly before 2023)": "#d68910", "accelerated trend": "#e67e22", "no change": "#999999"}
VCOL = {"MLSys": "#1f77b4", "OSDI": "#ff7f0e", "NSDI": "#2ca02c", "VLDB": "#d62728", "NeurIPS": "#9467bd",
        "ICML": "#8c564b", "CVPR": "#e377c2", "ACL": "#17becf", "ICLR": "#d62728", "AAAI": "#393b79", "COLM*": "#7f7f7f", "EuroSys*": "#bcbd22"}
lab = lambda f: LABEL.get(f, f)

# 1) Effect-size forest: pooled AUC with per-venue dots.
pool = rd("pooled.csv")
pv = rd("per_venue.csv")
top = [p for p in pool if p["feature"] not in ("words",)][:30][::-1]
fig, ax = plt.subplots(figsize=(7.2, 8.5))
for i, p in enumerate(top):
    f = p["feature"]
    for r in pv:
        if r["feature"] == f and not r["venue"].endswith("*"):
            ax.scatter(fl(r["auc"]), i, s=10, color=VCOL[r["venue"]], alpha=0.7, zorder=2)
    ax.scatter(fl(p["mean_auc"]), i, marker="D", s=36, color=COL.get(p["attribution"], "k"), zorder=3,
               edgecolor="k", linewidth=0.5)
    if not math.isnan(fl(p["trend_mean_auc"])):
        ax.scatter(fl(p["trend_mean_auc"]), i, marker="|", s=80, color="k", zorder=3)
ax.axvline(0.5, color="#555", lw=0.8, ls="--")
ax.set_yticks(range(len(top)))
ax.set_yticklabels([lab(p["feature"]) for p in top], fontsize=7.5)
ax.set_xlabel("AUC, pre-LLM (2019-20) vs post-GPT-4 (2024-26)   [| = AUC 2016 vs 2020, the pre-LLM trend]")
ax.set_xlim(0.1, 1.0)
from matplotlib.lines import Line2D  # noqa: E402
h = [Line2D([], [], marker="D", ls="", color=c, mec="k", label=k.replace("AI-era", "LLM-era").replace("pre-AI", "pre-LLM")) for k, c in COL.items()]
h += [Line2D([], [], marker="o", ls="", color=c, ms=4, label=k) for k, c in VCOL.items() if not k.endswith("*")]
fig.legend(handles=h, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 0.005),
           ncol=4, frameon=False)
fig.savefig(os.path.join(F, "fig1_effects.png"))
plt.close(fig)

# 2) Yearly series for ACL and ICML, normalised to 2020.
ser = rd("series.csv")
feats = ["ing_interp_core_1k", "ing_func_1k", "boost_1k", "op_additionally_1k", "style_lex_1k",
         "emdash_1k", "llm_1k", "para_len"]
fig, axes = plt.subplots(2, 4, figsize=(10, 4.6), sharex=True)
for ax, f in zip(axes.flat, feats):
    for r in ser:
        if r["feature"] == f:
            ys = [fl(r[str(y)]) for y in range(2020, 2026)]
            ax.plot(range(2020, 2026), ys, marker="o", ms=3, label=r["venue"], color=VCOL[r["venue"]])
    ax.axvspan(2022.9, 2025.2, color="#f5b7b1", alpha=0.25, lw=0)
    ax.set_title(lab(f), fontsize=8)
    ax.set_xticks(range(2020, 2026))
    ax.set_xticklabels(["20", "21", "22", "23", "24", "25"])
axes[0, 0].legend(fontsize=7, frameon=False)
fig.suptitle("Ten venues, one sample per year; shaded = published after ChatGPT (Nov 2022)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(F, "fig2_series.png"))
plt.close(fig)

# 3) Style-index distribution per corpus (box plots), and share above the pre-AI p95.
si = rd("style_index.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
xs, labels = [], []
for i, r in enumerate(si):
    x = i
    col = VCOL[r["venue"]]
    ax1.plot([x, x], [fl(r["p10"]), fl(r["p90"])], color=col, lw=1)
    ax1.add_patch(plt.Rectangle((x - 0.3, fl(r["p25"])), 0.6, fl(r["p75"]) - fl(r["p25"]), color=col,
                                alpha=0.35 if r["year"] < "2023" else 0.85))
    ax1.plot([x - 0.3, x + 0.3], [fl(r["p50"])] * 2, color="k", lw=1)
    xs.append(x)
    labels.append(r["corpus"].replace("20", "'", 1))
ax1.set_xticks(xs)
ax1.set_xticklabels(labels, rotation=90, fontsize=6.5)
ax1.axhline(0, color="#555", lw=0.6, ls="--")
ax1.set_ylabel("new-style index (mean z vs venue's pre-LLM papers)")
ax1.set_title("Distribution per corpus (box = p25-p75, line = p10-p90)", fontsize=8)
for v in ("ACL", "ICML"):
    rs = sorted([r for r in si if r["venue"] == v], key=lambda r: r["year"])
    ax2.plot([int(r["year"]) for r in rs], [fl(r["share_above_pre_p95"]) * 100 for r in rs], marker="o",
             label=v, color=VCOL[v])
for v in ("MLSys", "OSDI", "NSDI", "VLDB", "NeurIPS", "CVPR"):
    rs = sorted([r for r in si if r["venue"] == v], key=lambda r: r["year"])
    ax2.plot([int(r["year"]) for r in rs], [fl(r["share_above_pre_p95"]) * 100 for r in rs], marker=".",
             lw=0.8, ls=":", label=v, color=VCOL[v])
ax2.axhline(5, color="#555", lw=0.6, ls="--")
ax2.axvspan(2022.9, 2026.2, color="#f5b7b1", alpha=0.25, lw=0)
ax2.set_ylabel("% of papers above the pre-LLM 95th percentile")
ax2.legend(fontsize=6.5, frameon=False, ncol=2)
ax2.set_title("Tail share by year (5% expected with no change)", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(F, "fig3_style_index.png"))
plt.close(fig)

# 4) Vocabulary: rise in all papers vs rise in non-LLM papers.
_v = rd("vocab.csv")
_maxv = max(int(r["venues"]) for r in _v)
vc = [r for r in _v if fl(r["post_df"]) >= 0.08 and int(r["venues"]) >= _maxv - 2]
fig, ax = plt.subplots(figsize=(7, 5.5))
xa = [fl(r["mean_lr"]) for r in vc]
ya = [fl(r["mean_lr_nonllm"]) for r in vc]
ax.scatter(xa, ya, s=3, color="#bbb")
# Label the 14 strongest style risers (rise as much without LLM papers) and 10 topic risers.
style = sorted([r for r in vc if " " not in r["term"] and fl(r["mean_lr_nonllm"]) > 0.85 * fl(r["mean_lr"])
                and not re.search(r"llm|gpt|llama|prompt|diffusion|model|transformer|token|rtx|sota", r["term"])],
               key=lambda r: -fl(r["mean_lr_nonllm"]))[:14]
topic = sorted([r for r in vc if " " not in r["term"] and fl(r["mean_lr_nonllm"]) < 0.6 * fl(r["mean_lr"])],
               key=lambda r: -fl(r["mean_lr"]))[:10]
for r in style:
    ax.scatter(fl(r["mean_lr"]), fl(r["mean_lr_nonllm"]), s=10, color="#c0392b")
ax.text(-1.2, 3.9, "selected risers also seen below LLM-topic threshold:\n" + "\n".join(
    f"  {r['term']}  ({fl(r['pre_df']) * 100:.0f}% → {fl(r['post_df']) * 100:.0f}% of papers)" for r in style),
        fontsize=6.5, va="top", color="#c0392b")
for r in topic:
    x, y = fl(r["mean_lr"]), fl(r["mean_lr_nonllm"])
    ax.scatter(x, y, s=10, color="#7d3c98")
    ax.annotate(r["term"], (x, y), fontsize=6.5, xytext=(3, -2), textcoords="offset points", color="#7d3c98")
lim = max(xa) + 0.2
ax.plot([0, lim], [0, lim], color="#555", lw=0.6, ls="--")
ax.set_xlabel("log ratio of document frequency, post vs pre (mean over 10 venues)")
ax.set_ylabel("same, using only non-LLM post papers")
ax.set_title("Change in all papers vs papers below the LLM-topic threshold", fontsize=8)
fig.savefig(os.path.join(F, "fig4_vocab.png"))
plt.close(fig)

# 5) Topic shift + intro openers.
ts = rd("topic_share.csv")
op = rd("intro_openers.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
ax1.bar(range(len(ts)), [fl(r["llm_topic_share"]) * 100 for r in ts],
        color=["#c0392b" if int(r["corpus"][-4:]) >= 2023 else "#2471a3" for r in ts])
ax1.set_xticks(range(len(ts)))
ax1.set_xticklabels([r["corpus"] for r in ts], rotation=90, fontsize=6)
ax1.set_ylabel("% LLM-centred papers (≥5 LLM mentions)")
cats = ["rise-of-X", "recently", "llm-subject", "has-become", "many-X", "definition", "other"]
ccol = ["#c0392b", "#e67e22", "#8e44ad", "#f1c40f", "#16a085", "#2980b9", "#d5d8dc"]
rows = [r for r in op if r["era"] in ("pre", "post") and not r["venue"].endswith("*")]
bottom = [0.0] * len(rows)
for c, col in zip(cats, ccol):
    vals = [fl(r[c]) * 100 for r in rows]
    ax2.bar(range(len(rows)), vals, bottom=bottom, color=col, label=c, width=0.8)
    bottom = [b + v for b, v in zip(bottom, vals)]
ax2.set_xticks(range(len(rows)))
ax2.set_xticklabels([f"{r['venue']} {r['era']}" for r in rows], rotation=90, fontsize=6.5)
ax2.set_ylabel("% of introductions")
ax2.legend(fontsize=6, frameon=False, bbox_to_anchor=(1.0, 1.0))
ax2.set_title("First sentence of the introduction", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(F, "fig5_topic_openers.png"))
plt.close(fig)
print("figures written to", F)

# 6) Plain -> elevated synonym pairs, and the excess-vocabulary curve by year.
if os.path.exists(os.path.join(R, "synonym_pairs_summary.csv")) and os.path.exists(os.path.join(R, "excess.csv")):
    sp = rd("synonym_pairs_summary.csv")
    ex = rd("excess.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    sp = sorted(sp, key=lambda r: fl(r["post"]) - fl(r["pre"]))
    for i, r in enumerate(sp):
        a, b, c = fl(r["2016"]) * 100, fl(r["pre"]) * 100, fl(r["post"]) * 100
        ax1.plot([b, c], [i, i], color="#bbb", lw=2, zorder=1)
        ax1.scatter(a, i, marker="|", s=80, color="k", zorder=3)
        ax1.scatter(b, i, s=22, color="#2471a3", zorder=2)
        ax1.scatter(c, i, s=22, color="#c0392b", zorder=2)
        ax1.scatter(fl(r["post_nonllm"]) * 100, i, s=22, facecolor="none", edgecolor="#c0392b", zorder=2)
    ax1.set_yticks(range(len(sp)))
    ax1.set_yticklabels([r["pair"] for r in sp], fontsize=7)
    ax1.set_xlabel("elevated variant, % of (plain + elevated) occurrences")
    ax1.set_title("| 2016   ● 2019-20   ● 2024-26   ○ 2024-26 non-LLM papers", fontsize=8)
    for v, col in (("ACL", "#17becf"), ("ICML", "#8c564b")):
        rs = sorted([r for r in ex if r["venue"] == v and r["corpus"].startswith(v.lower())], key=lambda r: r["corpus"])
        seen = {}
        for r in rs:
            seen[int(r["corpus"][-4:])] = fl(r["share_3plus_markers"]) * 100
        ys = sorted(seen)
        ax2.plot(ys, [seen[y] for y in ys], marker="o", color=col, label=v)
    for v in ("MLSys", "OSDI", "NSDI", "VLDB", "NeurIPS", "CVPR"):
        rs = [r for r in ex if r["venue"] == v]
        pts = sorted({int(r["corpus"][-4:]): fl(r["share_3plus_markers"]) * 100 for r in rs}.items())
        ax2.plot([p[0] for p in pts], [p[1] for p in pts], marker=".", ls=":", lw=0.8, color=VCOL[v], label=v)
    ax2.axvspan(2022.9, 2026.2, color="#f5b7b1", alpha=0.25, lw=0)
    ax2.set_ylabel("% of papers with ≥3 marker-word uses")
    ax2.set_title("Marker words (underscore, showcase, align with, overlook, ...)", fontsize=8)
    ax2.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig6_pairs_excess.png"))
    plt.close(fig)
