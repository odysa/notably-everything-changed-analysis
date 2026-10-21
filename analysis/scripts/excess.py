#!/usr/bin/env python3
"""Excess-vocabulary lower bound (after Kobak et al. 2024), with split-half marker selection.

Usage: python3 -I excess.py <workdir> <outdir>   -> <outdir>/excess.csv, <outdir>/excess_markers.json

Papers are split into two halves by a hash of their id. Markers are selected on half A only:
single words whose document frequency rises from the pre-LLM to the post-GPT-4 sample in every
venue, rise about as much in non-LLM papers, did not rise 2016 -> 2020, and are not topic words.
The share of papers using >= 3 markers is then measured on half B only, so selection cannot
inflate the estimate (no winner's curse). If p papers used markers before and q do now, at least
q - p changed their wording (assuming no other cause).
"""
import csv, hashlib, json, math, os, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import VENUES, SERIES, mean  # noqa: E402
from features import WORD  # noqa: E402

N_MARKERS = 16
TOPIC = re.compile(r"llm|gpt|llama|qwen|prompt|token|diffusion|transformer|openai|foundation|vllm|reasoning|agent|"
                   r"multimodal|mllm|lora|rlhf|instruction|fine-tun|deepseek|gemini|claude|mistral|chatbot|chat|sota|"
                   r"benchmark|alignment|hallucinat|kv|moe|rag|retrieval|generative|inference|serving|accelerat|gpu|h100|a100")
STOP = set("the of and to in a for is that we this on with as by are be it which from an can our not or at these each".split())
half = lambda pid: int(hashlib.md5(pid.encode()).hexdigest(), 16) % 2


def main():
    wd, out = sys.argv[1], sys.argv[2]
    meta = {}
    for line in open(os.path.join(wd, "features.jsonl")):
        p = json.loads(line)
        meta[(p["corpus"], p["id"])] = p["llm_topic"]
    v8 = [v for v in VENUES if not v.endswith("*")]
    corpora = sorted({c for v in VENUES.values() for grp in v for c in grp} | {c for s in SERIES.values() for c in s}
                     | {re.sub(r"\d{4}$", str(y), s[0]) for s in SERIES.values() for y in (2017, 2018, 2019)})
    sets = {}
    for c in corpora:
        d = os.path.join(wd, "text", c)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            rec = json.load(open(os.path.join(d, fn)))
            if (c, rec["id"]) in meta:
                toks = [w.lower() for w in WORD.findall(" ".join(rec["paras"]))]
                sets[(c, rec["id"])] = (Counter(t for t in toks if len(t) >= 4 and t not in STOP), meta[(c, rec["id"])])
    # ---- selection on half A ----
    df = defaultdict(Counter)
    n = Counter()
    for (c, pid), (cnt, llm) in sets.items():
        if half(pid) != 0:
            continue
        for key in (c, c + "|nl") if not llm else (c,):
            df[key].update(cnt.keys())
            n[key] += 1
    rate = lambda cs, t: (sum(df[c][t] for c in cs) + 0.5) / (sum(n[c] for c in cs) + 1)
    vocab = {t for c in df for t, k in df[c].items() if k >= 5}
    cands = []
    for t in vocab:
        if TOPIC.search(t):
            continue
        lrs, lnl, tr, posts = [], [], [], []
        for v in v8:
            ctrl, pre, post = VENUES[v]
            a, b = rate(pre, t), rate(post, t)
            lrs.append(math.log(b / a))
            lnl.append(math.log(rate([c + "|nl" for c in post], t) / a))
            posts.append(b)
            if ctrl:
                tr.append(math.log(a / rate(ctrl, t)))
        if min(lrs) > 0.2 and mean(posts) >= 0.05 and mean(lnl) >= 0.8 * mean(lrs) and (not tr or mean(tr) < 0.3):
            cands.append((mean(lrs), t))
    markers = [t for _, t in sorted(cands, reverse=True)[:N_MARKERS]]
    json.dump({"markers": markers, "selection": "half A (md5(id) even)", "evaluation": "half B"},
              open(os.path.join(out, "excess_markers.json"), "w"), indent=1)
    # ---- evaluation on half B ----
    by_corpus = defaultdict(list)
    for (c, pid), (cnt, llm) in sets.items():
        if half(pid) == 1:
            by_corpus[c].append((sum(cnt[m] for m in markers), llm))
    rows = []
    groups = list(VENUES.items()) + [(k, ([], [cs[0]], cs[1:])) for k, cs in SERIES.items()]
    for v, (ctrl, pre, post) in groups:
        base = mean([mean([h[0] >= 3 for h in by_corpus[c]]) for c in pre if by_corpus.get(c)])
        extra = [re.sub(r"\d{4}$", str(y), pre[0]) for y in (2017, 2018, 2019)
                 if re.sub(r"\d{4}$", str(y), pre[0]) not in pre] if v in SERIES else []
        for c in ctrl + extra + pre + post:
            hits = by_corpus.get(c)
            if not hits:
                continue
            nl = [h for h in hits if not h[1]]
            rows.append({"venue": v, "corpus": c, "n": len(hits), "share_any_marker": mean([h[0] >= 1 for h in hits]),
                         "share_3plus_markers": mean([h[0] >= 3 for h in hits]),
                         "share_3plus_nonllm": mean([h[0] >= 3 for h in nl]) if nl else float("nan"),
                         "excess_vs_pre": mean([h[0] >= 3 for h in hits]) - base})
    with open(os.path.join(out, "excess.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(x, 3) if isinstance(x, float) else x for k, x in r.items()})
    print("markers (selected on half A):", ", ".join(markers))
    for r in rows:
        print(r["venue"], r["corpus"], r["n"], f"3+={r['share_3plus_markers']:.2f} excess={r['excess_vs_pre']:+.2f}")


if __name__ == "__main__":
    main()
