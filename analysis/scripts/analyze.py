#!/usr/bin/env python3
"""Compare pre-AI (2019-2020) vs post-ChatGPT (2024-2026) papers per venue, with a 2016 trend
control, a 2020-2025 yearly series (ACL, ICML) and a topic control (non-LLM papers only).

Usage: python3 -I analyze.py <workdir> <outdir>
Reads <workdir>/text/<corpus>/*.json (from extract.py); writes CSV/JSON/Markdown to <outdir>.
"""
import csv, json, math, os, random, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import paper_features, WORD, sentences  # noqa: E402

# venue -> (control corpora, pre corpora, post corpora)
VENUES = {
    "MLSys":   ([], ["mlsys2019", "mlsys2020"], ["mlsys2025", "mlsys2026"]),
    "OSDI":    (["osdi2016"], ["osdi2020"], ["osdi2025", "osdi2026"]),
    "NSDI":    (["nsdi2016"], ["nsdi2020"], ["nsdi2025", "nsdi2026"]),
    "VLDB":    ([], ["vldb2020"], ["vldb2025", "vldb2026"]),
    "NeurIPS": (["neurips2016"], ["neurips2020"], ["neurips2024"]),
    "ICML":    (["icml2016"], ["icml2020"], ["icml2025", "icml2026"]),
    "CVPR":    (["cvpr2016"], ["cvpr2020"], ["cvpr2025", "cvpr2026"]),
    "ACL":     (["acl2016"], ["acl2020"], ["acl2025", "acl2026"]),
    # ICLR: accepted papers found on arXiv by their "ICLR <year>" comment (OpenReview blocks scripts).
    "ICLR":    (["iclr2016"], ["iclr2020"], ["iclr2025", "iclr2026"]),
    "AAAI":    (["aaai2016"], ["aaai2020"], ["aaai2025", "aaai2026"]),
    # COLM began in 2024: no pre-AI year. Baseline = the closest 2020 LM/ML venues.
    "COLM*":   ([], ["acl2020", "neurips2020"], ["colm2024", "colm2025", "colm2026"]),
    # EuroSys: only arXiv versions of 2025-26 papers; baseline = OSDI + NSDI 2020.
    "EuroSys*": ([], ["osdi2020", "nsdi2020"], ["eurosys2025"]),
}
# Yearly series: pre-LLM year, the transition years 2021-2024 (sampled in round 2 for every venue
# but NeurIPS 2024, which is NeurIPS's post sample), and the first post-GPT-4 year.
SERIES = {"ACL": ["acl2020", "acl2021", "acl2022", "acl2023", "acl2024", "acl2025"],
          "ICML": ["icml2020", "icml2021", "icml2022", "icml2023", "icml2024", "icml2025"],
          "OSDI": ["osdi2020", "osdi2021", "osdi2022", "osdi2023", "osdi2024", "osdi2025"],
          "NSDI": ["nsdi2020", "nsdi2021", "nsdi2022", "nsdi2023", "nsdi2024", "nsdi2025"],
          "MLSys": ["mlsys2020", "mlsys2021", "mlsys2022", "mlsys2023", "mlsys2024", "mlsys2025"],
          "VLDB": ["vldb2020", "vldb2021", "vldb2022", "vldb2023", "vldb2024", "vldb2025"],
          "CVPR": ["cvpr2020", "cvpr2021", "cvpr2022", "cvpr2023", "cvpr2024", "cvpr2025"],
          "NeurIPS": ["neurips2020", "neurips2021", "neurips2022", "neurips2023", "neurips2024"],
          "ICLR": ["iclr2020", "iclr2021", "iclr2022", "iclr2023", "iclr2024", "iclr2025"],
          "AAAI": ["aaai2020", "aaai2021", "aaai2022", "aaai2023", "aaai2024", "aaai2025"]}
YEAR = lambda c: int(re.search(r"\d{4}", c).group())

NUMERIC_SKIP = {"intro_open", "intro_first", "llm_topic"}
STRUCT = ["words", "pages", "n_sections", "intro_words", "related_words", "has_related", "related_early",
          "has_limitations", "limitations_words", "has_discussion", "discussion_words", "has_ethics_impact",
          "conclusion_words", "has_future_work", "contrib_list", "contrib_bullets", "n_figs", "n_tabs",
          "n_refs", "github", "artifact", "has_appendix", "abstract_words", "abstract_has_quant"]


def load(wd):
    cache = os.path.join(wd, "features.jsonl")
    papers = []
    if os.path.exists(cache) and os.path.getmtime(cache) > max(
            os.path.getmtime(os.path.join(wd, "text", c)) for c in os.listdir(os.path.join(wd, "text"))):
        return [json.loads(l) for l in open(cache)]
    for c in sorted(os.listdir(os.path.join(wd, "text"))):
        for fn in sorted(os.listdir(os.path.join(wd, "text", c))):
            rec = json.load(open(os.path.join(wd, "text", c, fn)))
            f = paper_features(rec)
            if f is None:
                continue
            # ACL 2020 "acl-main" mixes short (4-page) and long papers: keep long ones.
            if c == "acl2020" and (f["pages"] < 8 or f["words"] < 3000):
                continue
            f["corpus"], f["id"] = c, rec["id"]
            papers.append(f)
    with open(cache, "w") as out:
        for p in papers:
            out.write(json.dumps(p) + "\n")
    return papers


def auc(a, b):
    """P(post > pre) + 0.5 P(tie): Mann-Whitney AUC; 0.5 = no separation."""
    if not a or not b:
        return float("nan")
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = {}, 0
    r = [0.0] * len(allv)
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        for k in range(i, j):
            r[k] = (i + j + 1) / 2
        i = j
    rb = sum(r[k] for k in range(len(allv)) if allv[k][1] == 1)
    nb, na = len(b), len(a)
    return (rb - nb * (nb + 1) / 2) / (na * nb)


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def share(x):
    return mean([1.0 if v > 0 else 0.0 for v in x])


def lr(post, pre, eps):
    return math.log((post + eps) / (pre + eps))


def boot_ci(a, b, stat, n=400, seed=0):
    rnd = random.Random(seed)
    vals = []
    for _ in range(n):
        aa = [rnd.choice(a) for _ in a]
        bb = [rnd.choice(b) for _ in b]
        vals.append(stat(aa, bb))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def group(papers, corpora, nonllm=False):
    return [p for p in papers if p["corpus"] in corpora and (not nonllm or not p["llm_topic"])]


def feature_names(papers):
    return [k for k, v in papers[0].items() if isinstance(v, (int, float)) and k not in NUMERIC_SKIP]


def per_venue_table(papers, feats):
    rows = []
    for v, (ctrl, pre, post) in VENUES.items():
        C, A, B = group(papers, ctrl), group(papers, pre), group(papers, post)
        Bn = group(papers, post, nonllm=True)
        if not A or not B:
            continue
        for f in feats:
            a, b, c, bn = [p[f] for p in A], [p[f] for p in B], [p[f] for p in C], [p[f] for p in Bn]
            eps = 0.05 * (mean(a + b) or 1e-3) + 1e-6
            yrs_pre = mean([YEAR(x) for x in pre]) - (mean([YEAR(x) for x in ctrl]) if ctrl else 0)
            yrs_post = mean([YEAR(x) for x in post]) - mean([YEAR(x) for x in pre])
            row = {"venue": v, "feature": f, "n_ctrl": len(c), "n_pre": len(a), "n_post": len(b),
                   "n_post_nonllm": len(bn),
                   "ctrl_mean": mean(c), "pre_mean": mean(a), "post_mean": mean(b), "post_nonllm_mean": mean(bn),
                   "ctrl_share": share(c), "pre_share": share(a), "post_share": share(b),
                   "log_ratio": lr(mean(b), mean(a), eps), "auc": auc(a, b),
                   "log_ratio_nonllm": lr(mean(bn), mean(a), eps) if bn else float("nan"),
                   "auc_nonllm": auc(a, bn) if bn else float("nan"),
                   "trend_log_ratio": lr(mean(a), mean(c), eps) if c else float("nan"),
                   "trend_auc": auc(c, a) if c else float("nan"),
                   # Annualised rates: change per year before AI vs after.
                   "rate_pre_per_yr": lr(mean(a), mean(c), eps) / yrs_pre if c else float("nan"),
                   "rate_post_per_yr": lr(mean(b), mean(a), eps) / yrs_post}
            rows.append(row)
    return rows


def pooled(rows, feats, exclude=("COLM*", "EuroSys*")):
    """Cross-venue summary: mean log ratio / AUC over venues, and how many venues agree."""
    out = []
    for f in feats:
        rs = [r for r in rows if r["feature"] == f and r["venue"] not in exclude]
        if not rs:
            continue
        ctrl = [r for r in rs if r["n_ctrl"]]
        lrs = [r["log_ratio"] for r in rs]
        up = sum(1 for r in rs if r["auc"] > 0.56)
        dn = sum(1 for r in rs if r["auc"] < 0.44)
        out.append({
            "feature": f, "venues": len(rs), "mean_log_ratio": mean(lrs),
            "mean_auc": mean([r["auc"] for r in rs]),
            "mean_auc_nonllm": mean([r["auc_nonllm"] for r in rs if not math.isnan(r["auc_nonllm"])]),
            "venues_up": up, "venues_down": dn,
            "pre_share": mean([r["pre_share"] for r in rs]), "post_share": mean([r["post_share"] for r in rs]),
            "pre_mean": mean([r["pre_mean"] for r in rs]), "post_mean": mean([r["post_mean"] for r in rs]),
            "trend_mean_log_ratio": mean([r["trend_log_ratio"] for r in ctrl]) if ctrl else float("nan"),
            "trend_mean_auc": mean([r["trend_auc"] for r in ctrl]) if ctrl else float("nan"),
            "rate_pre": mean([r["rate_pre_per_yr"] for r in ctrl]) if ctrl else float("nan"),
            "rate_post": mean([r["rate_post_per_yr"] for r in ctrl]) if ctrl else float("nan"),
            "rate_post_all": mean([r["rate_post_per_yr"] for r in rs]),
        })
    return out


def attribute(p, series_info):
    """Label a pooled change. Order matters:
    no change  -> |AUC - 0.5| < 0.06 and small log ratio;
    topic      -> the shift mostly vanishes in non-LLM papers;
    trend      -> already moving 2016->2020 at >= 60% of the post-2020 yearly rate; if the yearly
                  series still puts most of the change after 2022, call it an accelerated trend;
    gradual    -> the ACL/ICML series puts < 40% of the 2020->2025 change after 2022;
    AI-era break otherwise. Structural features driven by venue rules are relabelled by hand in
    the paper (pages, limitations, ethics statements)."""
    a = p["mean_auc"]
    if abs(a - 0.5) < 0.06 and abs(p["mean_log_ratio"]) < 0.2:
        return "no change"
    sign = 1 if a > 0.5 else -1
    if not math.isnan(p["mean_auc_nonllm"]) and abs(p["mean_auc_nonllm"] - 0.5) < 0.5 * abs(a - 0.5):
        return "topic"
    br = series_info.get(p["feature"], {}).get("break_share", float("nan"))
    rp, rq = p["rate_pre"], p["rate_post"]
    if not math.isnan(rp) and sign * rp > 0 and abs(rp) >= 0.6 * abs(rq):
        return "accelerated trend" if (not math.isnan(br) and br >= 0.6) else "pre-AI trend"
    if not math.isnan(br) and br < 0.4:
        return "gradual (mostly before 2023)"
    return "AI-era break"


def series_table(papers, feats):
    """Yearly means per venue; break_share = fraction of the first->last change that happens after
    2022 (between the 2021-22 mean and the mean of years >= 2024), averaged over venues."""
    out, info = [], {}
    for f in feats:
        shares = []
        for v, cs in SERIES.items():
            ys = {YEAR(c): mean([p[f] for p in group(papers, [c])]) for c in cs}
            ys = {y: m for y, m in ys.items() if not math.isnan(m)}
            out.append({"venue": v, "feature": f, **ys})
            if len(ys) < 4:
                continue
            first, last = ys[min(ys)], ys[max(ys)]
            early = [ys[y] for y in (2021, 2022) if y in ys]
            late = [ys[y] for y in ys if y >= 2024]
            total = last - first
            if not early or not late or abs(total) < 1e-9:
                continue
            shares.append(max(0.0, min(1.5, (mean(late) - mean(early)) / total)))
        if shares:
            info[f] = {"break_share": mean(shares)}
    return out, info


# ---------------- vocabulary ----------------
STOP = set("""a an the and or of to in for on with by from at as is are was were be been being this that these
those it its we our us they their them he she his her i you your can will would should could may might must
not no than then there here which who whom whose what when where why how all any each both few more most other
some such only own same so too very s t just also into over under about above below between through during
before after again further once do does did doing have has had having if while because until against up down
out off et al fig figure table section eq e g i e via""".split())


def doc_terms(paras, bigrams=True):
    toks = [w.lower() for w in WORD.findall(" ".join(paras))]
    s = {w for w in toks if len(w) >= 3 and w not in STOP}
    if bigrams:
        s |= {a + " " + b for a, b in zip(toks, toks[1:])
              if a not in STOP or b not in STOP if len(a) > 1 and len(b) > 1}
    return s


def vocab(wd, papers):
    keep = {(p["corpus"], p["id"]) for p in papers}
    llm = {(p["corpus"], p["id"]) for p in papers if p["llm_topic"]}
    df = defaultdict(Counter)       # corpus -> term df
    df_nl = defaultdict(Counter)    # corpus -> term df among non-LLM papers
    n, n_nl = Counter(), Counter()
    unis = {}
    for c in sorted(os.listdir(os.path.join(wd, "text"))):
        for fn in os.listdir(os.path.join(wd, "text", c)):
            rec = json.load(open(os.path.join(wd, "text", c, fn)))
            if (c, rec["id"]) not in keep:
                continue
            terms = doc_terms(rec["paras"])
            unis[(c, rec["id"])] = frozenset(t for t in terms if " " not in t)
            df[c].update(terms)
            n[c] += 1
            if (c, rec["id"]) not in llm:
                df_nl[c].update(terms)
                n_nl[c] += 1
    rate = lambda D, N, cs, t: (sum(D[c][t] for c in cs) + 0.5) / (sum(N[c] for c in cs) + 1)
    terms = set()
    for c in df:
        terms |= {t for t, k in df[c].items() if k >= 3}
    rows = []
    for t in terms:
        lrs, lrs_nl, trend, pres, posts = [], [], [], [], []
        for v, (ctrl, pre, post) in VENUES.items():
            if v.endswith("*") or not all(c in n for c in pre + post):
                continue
            a, b = rate(df, n, pre, t), rate(df, n, post, t)
            lrs.append(math.log(b / a))
            pres.append(a)
            posts.append(b)
            lrs_nl.append(math.log(rate(df_nl, n_nl, post, t) / a))
            if ctrl and all(c in n for c in ctrl):
                trend.append(math.log(a / rate(df, n, ctrl, t)))
        if not lrs:
            continue
        rows.append({"term": t, "mean_lr": mean(lrs), "mean_lr_nonllm": mean(lrs_nl),
                     "min_lr": min(lrs), "venues_up": sum(x > 0.4 for x in lrs), "venues": len(lrs),
                     "trend_lr": mean(trend) if trend else float("nan"),
                     "pre_df": mean(pres), "post_df": mean(posts)})
    return rows, unis, df, n


# Prespecified "new style" index: features chosen from the earlier 2021-vs-2025 MLSys/OSDI scan
# and the AI-writing literature, before this corpus was analysed.
STYLE_INDEX = ["ing_interp_core_1k", "ing_func_1k", "boost_1k", "op_additionally_1k", "op_notably_1k",
               "emdash_1k", "this_closer_1k", "style_lex_1k", "famous_ai_words_1k"]
ALL_ERAS = {**{c: v for v, (ctrl, pre, post) in VENUES.items() if not v.endswith("*") for c in ctrl + pre + post},
            **{c: v for v, cs in SERIES.items() for c in cs}, "colm2024": "COLM*", "colm2025": "COLM*", "colm2026": "COLM*",
            "eurosys2025": "EuroSys*",
            # Pre-LLM trend years between the 2016 control and the 2019-20 baseline.
            **{re.sub(r"\d{4}$", str(y), cs[0]): v for v, cs in SERIES.items() for y in (2017, 2018, 2019)
               if re.sub(r"\d{4}$", str(y), cs[0]) != "mlsys2019"}}


def cosine(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def mean_pairwise(items, sim, k=60, reps=20, seed=0):
    """Mean pairwise similarity on equal-size random subsets (k papers), averaged over reps."""
    rnd = random.Random(seed)
    if len(items) < 10:
        return float("nan")
    k = min(k, len(items))
    vals = []
    for _ in range(reps):
        sub = rnd.sample(items, k)
        tot = cnt = 0
        for i in range(k):
            for j in range(i + 1, k):
                tot += sim(sub[i], sub[j])
                cnt += 1
        vals.append(tot / cnt)
    return mean(vals)


def extras(papers, unis, df, n, out):
    # 1) Homogenization: are papers in an era more alike than papers in the baseline era?
    hom = []
    for v, (ctrl, pre, post) in VENUES.items():
        base = group(papers, ctrl + pre + post)
        if not group(papers, pre):
            continue
        # Scale each function word by its SD over all eras of the venue pooled, so no era is the
        # reference (scaling by the pre-AI SD alone makes the pre-AI group look tightest).
        dim = len(base[0]["fw"])
        mu = [mean([p["fw"][i] for p in base]) for i in range(dim)]
        sd = [max(1e-6, (mean([(p["fw"][i] - mu[i]) ** 2 for p in base])) ** 0.5) for i in range(dim)]
        # Scale by the baseline SD but do not centre: a shared shift of the whole era must not
        # read as "more alike". Dispersion = mean pairwise Euclidean distance (lower = more alike).
        z = lambda p: [p["fw"][i] / sd[i] for i in range(dim)]
        allc = ctrl + pre + post
        # Content words: unigrams with venue-level df between 3% and 60% (drops boilerplate + rare).
        N = sum(n[c] for c in allc if c in n)
        vocabset = {t for t in set().union(*[set(df[c]) for c in allc if c in df]) if " " not in t
                    and 0.03 <= sum(df[c][t] for c in allc if c in df) / N <= 0.6}
        jac = lambda a, b: len(a & b) / len(a | b) if a | b else 0.0
        for era, cs in (("2016", ctrl), ("pre", pre), ("post", post), ("post non-LLM", post)):
            g = group(papers, cs, nonllm=(era == "post non-LLM"))
            if not g:
                continue
            zs = [z(p) for p in g]
            sets = [unis[(p["corpus"], p["id"])] & vocabset for p in g if (p["corpus"], p["id"]) in unis]
            hom.append({"venue": v, "era": era, "n": len(g),
                        "style_dispersion": mean_pairwise(zs, lambda a, b: math.dist(a, b)),
                        "content_jaccard": mean_pairwise(sets, jac),
                        "mattr": mean([p["mattr"] for p in g])})
    write_csv(os.path.join(out, "homogeneity.csv"), hom)
    # 2) New-style index: mean z-score (vs the venue's pre-AI papers) over STYLE_INDEX features.
    stats = {}
    for v, (ctrl, pre, post) in VENUES.items():
        # Centre on the pre-AI mean; scale by the SD of pre+post pooled, so a feature that is
        # almost always zero before AI (e.g. "delve") cannot explode the index.
        base, both = group(papers, pre), group(papers, pre + post)
        if not base:
            continue
        stats[v] = {}
        for f in STYLE_INDEX:
            m_all = mean([p[f] for p in both])
            sd = mean([(p[f] - m_all) ** 2 for p in both]) ** 0.5
            stats[v][f] = (mean([p[f] for p in base]), max(sd, 1e-3))
    for p in papers:
        v = ALL_ERAS.get(p["corpus"])
        if v in stats:
            p["style_index"] = mean([(p[f] - stats[v][f][0]) / stats[v][f][1] for f in STYLE_INDEX])
    rows = []
    for v, (ctrl, pre, post) in VENUES.items():
        base = sorted(p["style_index"] for p in group(papers, pre))
        if not base:
            continue
        p95 = base[int(0.95 * (len(base) - 1))]
        cs = sorted({c for c, vv in ALL_ERAS.items() if vv == v}, key=YEAR)
        if v.endswith("*"):
            cs = post
        for c in cs:
            g = sorted(p["style_index"] for p in group(papers, [c]) if "style_index" in p)
            if not g:
                continue
            q = lambda x: g[int(x * (len(g) - 1))]
            g_nl = [p["style_index"] for p in group(papers, [c], nonllm=True) if "style_index" in p]
            g_l = [p["style_index"] for p in group(papers, [c]) if p["llm_topic"] and "style_index" in p]
            rows.append({"venue": v, "corpus": c, "year": YEAR(c), "n": len(g), "mean": mean(g),
                         "p10": q(.1), "p25": q(.25), "p50": q(.5), "p75": q(.75), "p90": q(.9),
                         "share_above_pre_p95": mean([x > p95 for x in g]),
                         "mean_nonllm": mean(g_nl), "mean_llm": mean(g_l), "n_llm": len(g_l)})
    write_csv(os.path.join(out, "style_index.csv"), rows)
    with open(os.path.join(out, "paper_scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        # Paper ids are hashed: the release is for corpus-level analysis, not per-paper rankings.
        w.writerow(["corpus", "paper_hash", "style_index", "llm_topic", "words"] + STYLE_INDEX)
        for p in papers:
            if "style_index" in p:
                h = __import__("hashlib").sha256((p["corpus"] + p["id"]).encode()).hexdigest()[:12]
                w.writerow([p["corpus"], h, round(p["style_index"], 3), p["llm_topic"], p["words"]]
                           + [round(p[f], 3) for f in STYLE_INDEX])


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    wd, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    papers = load(wd)
    feats = feature_names(papers)
    counts = Counter(p["corpus"] for p in papers)
    json.dump(dict(sorted(counts.items())), open(os.path.join(out, "corpus_counts.json"), "w"), indent=1)
    rows = per_venue_table(papers, feats)
    write_csv(os.path.join(out, "per_venue.csv"), rows)
    srows, sinfo = series_table(papers, feats)
    write_csv(os.path.join(out, "series.csv"), srows)
    pool = pooled(rows, feats)
    for p in pool:
        p["attribution"] = attribute(p, sinfo)
        p["break_share"] = sinfo.get(p["feature"], {}).get("break_share", float("nan"))
    pool.sort(key=lambda p: -abs(p["mean_auc"] - 0.5))
    write_csv(os.path.join(out, "pooled.csv"), pool)
    # Intro openers: category shares per venue and era.
    op = []
    for v, (ctrl, pre, post) in VENUES.items():
        for era, cs in (("2016", ctrl), ("pre", pre), ("post", post)):
            g = group(papers, cs)
            if not g:
                continue
            cnt = Counter(p["intro_open"] for p in g)
            op.append({"venue": v, "era": era, "n": len(g),
                       **{k: cnt[k] / len(g) for k in ("rise-of-X", "recently", "llm-subject", "has-become",
                                                       "many-X", "definition", "other", "none")}})
    write_csv(os.path.join(out, "intro_openers.csv"), op)
    # LLM topic share per corpus (non-AI driver).
    write_csv(os.path.join(out, "topic_share.csv"),
              [{"corpus": c, "n": counts[c], "llm_topic_share": mean([p["llm_topic"] for p in group(papers, [c])])}
               for c in sorted(counts)])
    vrows, unis, df, n = vocab(wd, papers)
    extras(papers, unis, df, n, out)
    vrows.sort(key=lambda r: -r["mean_lr"])
    write_csv(os.path.join(out, "vocab.csv"), vrows)
    print(json.dumps(dict(sorted(counts.items()))))
    print(f"{len(papers)} papers, {len(feats)} features, {len(vrows)} terms")


if __name__ == "__main__":
    main()
