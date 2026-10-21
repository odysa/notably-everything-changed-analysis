#!/usr/bin/env python3
"""Second batch of shifts: readability, word shape, voice, contribution naming, titles, abstract
openers, and the reference lists themselves.

Usage: python newfeat.py <workdir> <outdir>   (needs numpy, scipy, pandas, statsmodels)
Writes <workdir>/newfeatures.jsonl, <outdir>/new_pooled.csv, <outdir>/new_per_venue.csv,
<outdir>/new_series.csv, <outdir>/abstract_openers.csv, <outdir>/contrib_verbs.csv, <outdir>/titles_sample.csv.
"""
import json, os, re, subprocess, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats as S  # noqa: E402
from features import WORD, sentences, LLM  # noqa: E402

REF_HEAD = re.compile(r"^\s*(\d+\.?\s*)?(R ?EFERENCES|References|B ?IBLIOGRAPHY|Bibliography)\s*$", re.M)
NOMINAL = re.compile(r"\b[a-z]{3,}(?:tion|sion|ment|ness|ity|ance|ence|ism)s?\b")
PASSIVE = re.compile(r"\b(?:is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?(?:\w+ed|shown|given|done|made|known|seen|"
                     r"taken|chosen|written|built|found|set|run|kept|held|drawn|proven)\b")
ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,}[a-z]?s?\b")
HYPHEN = re.compile(r"\b[a-z]{2,}-[a-z]{2,}(?:-[a-z]{2,})?\b")
CONTRIB = re.compile(r"\bwe (propose|present|introduce|develop|design|build|describe)\b", re.I)
NAMING = re.compile(r"\bwe (?:propose|present|introduce|develop|design|build)\s+(?:[A-Z][\w-]*[A-Z][\w-]*|[A-Z]{2,}[\w-]*)"
                    r"(?:\s*\([^)]{1,40}\))?,\s+(?:a|an)\b")
NOVEL_A = re.compile(r"\ba novel\b", re.I)
EXTENSIVE = re.compile(r"\b(?:extensive|comprehensive) (?:experiments|evaluations?|experimental|empirical|analys[ie]s)\b", re.I)
FRAMEWORK = re.compile(r"\bframeworks?\b", re.I)
SKIP_TITLE = re.compile(r"Proceedings|USENIX|PVLDB|arXiv|Conference|Workshop|Association|Published|ISBN|Copyright|"
                        r"Volume|Vol\.|pp\.|^\d+$|Open access|This paper is included|Long Papers", re.I)


def syllables(w):
    w = w.lower()
    v = re.findall(r"[aeiouy]+", w)
    n = len(v) - (1 if w.endswith("e") and not w.endswith(("le", "ee")) and len(v) > 1 else 0)
    return max(1, n)


def title_of(path, page1):
    meta = subprocess.run(["pdfinfo", path], capture_output=True, text=True, errors="replace").stdout
    m = re.search(r"^Title:\s*(.+)$", meta, re.M)
    t = m.group(1).strip() if m else ""
    if len(t.split()) >= 3 and not re.search(r"\.(?:dvi|pdf|tex|docx?)$|Microsoft|untitled|^paper\d*$", t, re.I):
        return t, "meta"
    lines, got = [l.strip() for l in page1.splitlines()], []
    for l in lines[:40]:
        if not l:
            if got:
                break
            continue
        if SKIP_TITLE.search(l) and not got:
            continue
        if re.search(r"@|University|Institute|\d{4,}|[∗*†‡]", l) and got:
            break
        got.append(l)
        if len(got) == 3:
            break
    return " ".join(got), "text"


def one(args):
    corpus, pid, path, pub_year = args
    rec = json.load(open(path.replace("/pdfs/", "/text/").replace(".pdf", ".json")))
    text = "\n".join(rec["paras"])
    words = WORD.findall(text)
    if len(words) < 1500:
        return None
    k = 1000.0 / len(words)
    sents = [s for p in rec["paras"] for s in sentences(p)]
    syl = [syllables(w) for w in words]
    asl = len(words) / max(len(sents), 1)
    asw = sum(syl) / len(words)
    page1 = subprocess.run(["pdftotext", "-l", "1", path, "-"], capture_output=True, text=True, errors="replace").stdout
    full = subprocess.run(["pdftotext", path, "-"], capture_output=True, text=True, errors="replace").stdout
    title, tsrc = title_of(path, page1)
    m = list(REF_HEAD.finditer(full))
    refs = full[m[0].end():] if m else ""
    # Stop at an appendix after the references.
    ap = re.search(r"\n\s*(?:[A-Z]\.?\s+)?(?:A ?PPENDIX|Appendix|Supplementary)", refs)
    refs = refs[:ap.start()] if ap else refs[:60000]
    years = [int(y) for y in re.findall(r"\b(19[5-9]\d|20[0-2]\d)[a-z]?\b", refs) if int(y) <= pub_year]
    n_arxiv = len(re.findall(r"arXiv|CoRR|abs/\d{4}\.\d{4,5}", refs))
    n_entries = max(len(set(re.findall(r"^\s*\[(\d{1,3})\]", refs, re.M))), len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\.", refs)), 1)
    contrib = [v.lower() for v in CONTRIB.findall(text)]
    ab = rec.get("abstract", "")
    ab_first = sentences(ab)[:1]
    ab_first = ab_first[0] if ab_first else ""
    return {
        "corpus": corpus, "id": pid,
        "syll_per_word": asw, "pct_long_words": float(np.mean([s >= 3 for s in syl])),
        "word_len_chars": float(np.mean([len(w) for w in words])),
        "flesch": 206.835 - 1.015 * asl - 84.6 * asw, "fk_grade": 0.39 * asl + 11.8 * asw - 15.59,
        "nominal_1k": len(NOMINAL.findall(text)) * k, "passive_per_100s": 100 * sum(bool(PASSIVE.search(s)) for s in sents) / max(len(sents), 1),
        "acronym_1k": len(ACRONYM.findall(text)) * k, "hyphen_1k": len(HYPHEN.findall(text)) * k,
        "paren_1k": text.count("(") * k,
        "contrib_n": len(contrib), "contrib_introduce_share": (contrib.count("introduce") / len(contrib)) if contrib else np.nan,
        "contrib_propose_share": (contrib.count("propose") / len(contrib)) if contrib else np.nan,
        "names_system": int(bool(NAMING.search(text))), "a_novel": int(bool(NOVEL_A.search(text))),
        "extensive_exp": int(bool(EXTENSIVE.search(text))), "framework_1k": len(FRAMEWORK.findall(text)) * k,
        "title": title, "title_src": tsrc, "title_words": len(title.split()),
        "title_colon": int(":" in title), "title_name_colon": int(bool(re.match(r"^[A-Z][\w-]*[A-Z0-9][\w-]*\s*:", title) or re.match(r"^[A-Z][a-z]+[A-Z][\w-]*\s*:", title))),
        "title_question": int("?" in title),
        "abstract_first": ab_first[:300],
        "ref_n": n_entries, "ref_arxiv_share": min(n_arxiv / n_entries, 1.5),
        "ref_median_age": (pub_year - float(np.median(years))) if len(years) >= 5 else np.nan,
        "ref_share_last2y": float(np.mean([pub_year - y <= 2 for y in years])) if len(years) >= 5 else np.nan,
    }


def abstract_open(s):
    s = s.strip()
    if LLM.match(s) or re.match(r"(?:Large|Multimodal|Vision)[- ](?:language|Language)", s):
        return "LLM subject"
    if re.match(r"(?:In|Within) this (?:paper|work)|This (?:paper|work)", s):
        return "In this paper / This paper"
    if re.match(r"We (?:propose|present|introduce|develop|design|study|show|describe|investigate|consider)", s):
        return "We + verb"
    if re.match(r"(?:The )?(?:recent |rapid )?(?:advent|emergence|rise|success|advances?|progress|proliferation|growth)\b", s, re.I) \
            or re.match(r"(?:With|As) (?:the )?(?:rapid|recent|increasing|growing)", s, re.I) or re.match(r"(?:Recently|In recent years)", s):
        return "rise / recently"
    if re.search(r"\b(?:has|have) (?:become|emerged|shown|demonstrated|achieved|gained)\b", s[:150]):
        return "has become / demonstrated"
    if re.match(r"(?:Despite|Although|While|However)", s):
        return "Despite / While"
    return "other"


NEW = ["syll_per_word", "pct_long_words", "word_len_chars", "flesch", "fk_grade", "nominal_1k", "passive_per_100s",
       "acronym_1k", "hyphen_1k", "paren_1k", "contrib_introduce_share", "contrib_propose_share", "names_system",
       "a_novel", "extensive_exp", "framework_1k", "title_words", "title_colon", "title_name_colon", "title_question",
       "ref_arxiv_share", "ref_median_age", "ref_share_last2y"]


def main():
    wd, out = sys.argv[1], sys.argv[2]
    base = S.load(wd)
    cache = os.path.join(wd, "newfeatures.jsonl")
    if not os.path.exists(cache):
        jobs = [(r.corpus, r.id, os.path.join(wd, "pdfs", r.corpus, r.id + ".pdf"), int(r.year)) for r in base.itertuples()]
        with Pool(8) as p:
            res = [r for r in p.imap_unordered(one, jobs, chunksize=8) if r]
        with open(cache, "w") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
    nf = pd.DataFrame([json.loads(l) for l in open(cache)])
    df = base.merge(nf, on=["corpus", "id"], how="inner")
    df["abstract_open"] = df["abstract_first"].fillna("").map(abstract_open)
    # Pooled effects with the same machinery as stats.py.
    rows, per = [], []
    for f in NEW:
        aucs, a16, anl = [], [], []
        for v in S.MAIN:
            pre = df[(df.venue == v) & (df.era == "pre")][f].dropna()
            post = df[(df.venue == v) & (df.era == "post")][f].dropna()
            c16 = df[(df.venue == v) & (df.era == "2016")][f].dropna()
            pnl = df[(df.venue == v) & (df.era == "post") & (df.llm_topic == 0)][f].dropna()
            if len(pre) < 10 or len(post) < 10:
                continue
            au = S.auc(pre, post)
            aucs.append(au)
            if len(c16) >= 10:
                a16.append(S.auc(c16, pre))
            if len(pnl) >= 10:
                anl.append(S.auc(pre, pnl))
            per.append({"feature": f, "venue": v, "c2016": c16.mean() if len(c16) else np.nan, "pre": pre.mean(),
                        "post": post.mean(), "post_nonllm": pnl.mean(), "auc": au,
                        "p": S.stats.mannwhitneyu(post, pre).pvalue})
        ser = {}
        for y in range(2020, 2026):
            ser[y] = df[df.venue.isin(["ACL", "ICML"]) & (df.year == y)][f].mean()
        rows.append({"feature": f, "pooled_auc": np.mean(aucs), "venues_up": sum(a > 0.56 for a in aucs),
                     "venues_down": sum(a < 0.44 for a in aucs), "auc_2016_to_2020": np.mean(a16) if a16 else np.nan,
                     "auc_nonllm": np.mean(anl), **{f"acl_icml_{y}": v for y, v in ser.items()}})
    pv = pd.DataFrame(per)
    pv["q_bh"] = S.bh(pv["p"].fillna(1))
    pv.to_csv(os.path.join(out, "new_per_venue.csv"), index=False)
    # Trend-adjusted DiD and changepoint for each new feature.
    # Rows missing a feature (e.g. no contribution verb) are dropped per feature, not imputed.
    S.KEY = NEW
    did = S.part_b(df).set_index("feature")
    cp = S.part_c(df, B=100).set_index("feature")
    pool = pd.DataFrame(rows).set_index("feature")
    pool = pool.join(did[["excess_jump_sd", "excess_se", "excess_p"]]).join(cp[["best_tau", "delta_bic_vs_linear"]])
    pool.reset_index().sort_values("pooled_auc").to_csv(os.path.join(out, "new_pooled.csv"), index=False)
    # Abstract openers and contribution verbs by venue x era.
    ab = df[df.era.isin(["2016", "pre", "series", "post"])].groupby(["venue", "era"])["abstract_open"].value_counts(normalize=True).unstack(fill_value=0)
    ab.to_csv(os.path.join(out, "abstract_openers.csv"))
    yr = df[df.venue.isin(["ACL", "ICML"])].groupby("year")["abstract_open"].value_counts(normalize=True).unstack(fill_value=0)
    yr.to_csv(os.path.join(out, "abstract_openers_by_year.csv"))
    df[["corpus", "venue", "era", "year", "title", "title_src"]].sample(60, random_state=0).to_csv(
        os.path.join(out, "titles_sample.csv"), index=False)
    df.drop(columns=["abstract_first"]).to_csv(os.path.join(wd, "all_features.csv"), index=False)
    print(pool.sort_values("pooled_auc").round(3).to_string())


if __name__ == "__main__":
    main()
