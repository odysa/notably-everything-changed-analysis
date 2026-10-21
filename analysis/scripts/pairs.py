#!/usr/bin/env python3
"""Plain vs elevated synonym pairs: did 'deal with' become 'address', 'use' become 'leverage'?

Usage: python3 -I pairs.py <workdir> <outdir>   -> <outdir>/synonym_pairs.csv
For each pair, the elevated share = elevated / (plain + elevated) occurrences, per corpus
(all papers, then non-LLM papers only), and paper-level document frequency of each side.
"""
import csv, json, math, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import VENUES, SERIES, mean  # noqa: E402
from features import LLM  # noqa: E402

PAIRS = [  # (name, plain regex, elevated regex)
    ("use → leverage/utilize", r"\b(?:use|uses|used|using)\b", r"\b(?:leverag\w+|utiliz\w+)\b"),
    ("show → demonstrate/showcase", r"\b(?:show|shows|showed|shown|showing)\b", r"\b(?:demonstrat\w+|showcas\w+)\b"),
    ("deal with/handle → address/tackle", r"\b(?:deal(?:s|t)? with|dealing with|handl(?:e|es|ed|ing))\b", r"\b(?:address(?:es|ed|ing)?|tackl(?:e|es|ed|ing))\b"),
    ("look at/study → examine/investigate/delve", r"\b(?:look(?:s|ed|ing)? at|stud(?:y|ies|ied))\b", r"\b(?:examin\w+|investigat\w+|delv\w+|explor\w+)\b"),
    ("help → facilitate/foster", r"\b(?:help|helps|helped|helping)\b", r"\b(?:facilitat\w+|foster\w*)\b"),
    ("improve → enhance/boost", r"\b(?:improv(?:e|es|ed|ing))\b", r"\b(?:enhanc(?:e|es|ed|ing)|boost(?:s|ed|ing)?)\b"),
    ("get → obtain/achieve", r"\b(?:get|gets|got|getting)\b", r"\b(?:obtain\w*|achiev\w+)\b"),
    ("important → crucial/critical/pivotal", r"\bimportant\b", r"\b(?:crucial|critical|pivotal|vital|paramount)\b"),
    ("big/large → substantial/significant", r"\b(?:big|large)\b", r"\b(?:substantial|significant)\b"),
    ("many/various → diverse", r"\b(?:many|various)\b", r"\bdiverse\b"),
    ("fail to/cannot → struggle to/fall short", r"\b(?:fail(?:s|ed)? to|cannot)\b", r"\b(?:struggl\w+|fall(?:s)? short)\b"),
    ("match/fit → align with", r"\b(?:match(?:es|ed)? (?:with|the)|fits?)\b", r"\balign(?:s|ed)? with\b"),
    ("ignore/miss → overlook", r"\b(?:ignor(?:e|es|ed|ing)|miss(?:es|ed|ing)?)\b", r"\boverlook\w*\b"),
    ("need → necessitate/require", r"\b(?:need|needs|needed)\b", r"\b(?:necessitat\w+|requir(?:e|es|ed|ing))\b"),
    ("we show → this demonstrates/highlights/underscores", r"\bwe (?:show|find|observe|see)\b",
     r"\b(?:this|these|which) (?:demonstrates?|highlights?|underscores?|showcases?|reveals?)\b"),
]
RX = [(n, re.compile(a, re.I), re.compile(b, re.I)) for n, a, b in PAIRS]


def main():
    wd, out = sys.argv[1], sys.argv[2]
    keep = {}
    for line in open(os.path.join(wd, "features.jsonl")):
        p = json.loads(line)
        keep[(p["corpus"], p["id"])] = p["llm_topic"]
    corpora = sorted({c for v in VENUES.values() for grp in v for c in grp} | {c for s in SERIES.values() for c in s})
    rows = []
    for c in corpora:
        d = os.path.join(wd, "text", c)
        if not os.path.isdir(d):
            continue
        acc = defaultdict(lambda: [0, 0, 0, 0, 0, 0])  # plain, elev, plain_nl, elev_nl, docs_elev, docs
        for fn in os.listdir(d):
            rec = json.load(open(os.path.join(d, fn)))
            if (c, rec["id"]) not in keep:
                continue
            nl = not keep[(c, rec["id"])]
            text = " ".join(rec["paras"])
            for n, a, b in RX:
                pa, pb = len(a.findall(text)), len(b.findall(text))
                r = acc[n]
                r[0] += pa
                r[1] += pb
                if nl:
                    r[2] += pa
                    r[3] += pb
                r[4] += pb > pa
                r[5] += 1
        for n, _, _ in RX:
            r = acc[n]
            rows.append({"corpus": c, "pair": n, "elevated_share": r[1] / max(1, r[0] + r[1]),
                         "elevated_share_nonllm": r[3] / max(1, r[2] + r[3]) if r[2] + r[3] else "",
                         "papers_preferring_elevated": r[4] / max(1, r[5]), "n": r[5]})
    with open(os.path.join(out, "synonym_pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    # Compact summary: pooled pre vs post over the 8 venues with a pre-AI year (mean of venues).
    summ = []
    for n, _, _ in RX:
        get1 = lambda cs, key: mean([float(r[key]) for r in rows if r["pair"] == n and r["corpus"] in cs and r[key] != ""])
        v8 = [v for v in VENUES if not v.endswith("*")]
        # Average within each venue first, then across venues, so venues with two post corpora
        # (OSDI, NSDI, MLSys) do not get double weight.
        get = lambda era, key: mean([x for x in (get1(VENUES[v][era], key) for v in v8 if VENUES[v][era]) if not math.isnan(x)])
        summ.append({"pair": n, "2016": get(0, "elevated_share"), "pre": get(1, "elevated_share"),
                     "post": get(2, "elevated_share"), "post_nonllm": get(2, "elevated_share_nonllm"),
                     **{f"acl{y}": get1([f"acl{y}"], "elevated_share") for y in range(2020, 2026)}})
    with open(os.path.join(out, "synonym_pairs_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(summ[0]))
        w.writeheader()
        for r in summ:
            w.writerow({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    for r in summ:
        print(r["pair"], "|", " ".join(f"{k}={v:.2f}" for k, v in r.items() if k != "pair"))


if __name__ == "__main__":
    main()
