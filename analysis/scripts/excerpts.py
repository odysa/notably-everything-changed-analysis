#!/usr/bin/env python3
"""Print candidate before/after sentences for each pattern, for hand-picking excerpt pairs.

Usage: python3 -I excerpts.py <workdir> [pattern-name ...] [--n 6] [--seed 0]
Picks random matching sentences from pre-AI corpora (2019-2020) and post corpora (2024-2026)
of the same venue family. Output: venue, era, paper id, sentence (trimmed to 320 chars).
"""
import json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import sentences, INTERP_CORE, THIS_CLOSER, LLM  # noqa: E402
from analyze import VENUES  # noqa: E402

PATTERNS = {
    "interp_rider": re.compile(r",\s+(?:thereby |thus |further )?(" + "|".join(INTERP_CORE) + r")\b"),
    "func_rider": re.compile(r",\s+(?:thereby |thus )?(enabling|ensuring|allowing|achieving|yielding|resulting in)\b"),
    "this_closer": THIS_CLOSER,
    "additionally": re.compile(r"^(Additionally|Notably|Crucially|Importantly),"),
    "emdash": re.compile(r"\w—\w"),
    "rather_than": re.compile(r"\brather than\b"),
    "boost": re.compile(r"\b(significantly|substantially|dramatically|remarkably|notably)\b"),
    "not_only": re.compile(r"\bnot (?:only|just|merely)\b"),
    "plain": re.compile(r"\b(deal(?:s)? with|keep(?:s)? track of|take advantage of|look at|is useful|well known)\b"),
    "elevated": re.compile(r"\b(aligns with|overlook\w*|underscor\w+|seamless\w*|intricate|struggles? to|necessitat\w+)\b"),
    "we_also": re.compile(r"^(We also|In addition,|Also,)"),
}
STYLE_WORDS = re.compile(r"\b(critical|crucial|pivotal|seamless(?:ly)?|holistic|intricate|underscore[sd]?|"
                         r"landscape|robust|comprehensive|nuanced|paradigm|cornerstone|fundamentally|key|"
                         r"leverag\w+|capabilit\w+)\b", re.I)


def main():
    wd = sys.argv[1]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 6
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    names = [a for a in sys.argv[2:] if a in PATTERNS or a in ("intro_first", "style_words")] or list(PATTERNS)
    rnd = random.Random(seed)
    eras = {}
    for v, (_, pre, post) in VENUES.items():
        if v.endswith("*"):
            continue
        for c in pre:
            eras[c] = (v, "pre")
        for c in post:
            eras[c] = (v, "post")
    pool = {k: {"pre": [], "post": []} for k in names}
    for c, (v, era) in eras.items():
        d = os.path.join(wd, "text", c)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            rec = json.load(open(os.path.join(d, fn)))
            sents = [s for p in rec["paras"] for s in sentences(p) if 8 <= len(s.split()) <= 60]
            for k in names:
                if k == "intro_first":
                    segs = [t for h, _, t in rec["segments"] if h.startswith("introduction") and t]
                    if segs:
                        first = sentences(re.sub(r"\s+", " ", segs[0]))[:1]
                        if first:
                            pool[k][era].append((v, rec["id"], first[0]))
                    continue
                rx = STYLE_WORDS if k == "style_words" else PATTERNS[k]
                for s in sents:
                    if rx.search(s):
                        pool[k][era].append((v, rec["id"], s))
    for k in names:
        print(f"\n######## {k}")
        for era in ("pre", "post"):
            items = pool[k][era]
            print(f"--- {era}: {len(items)} matches")
            for v, pid, s in rnd.sample(items, min(n, len(items))):
                print(f"[{v} {era} {pid}] {s[:320]}")


if __name__ == "__main__":
    main()
