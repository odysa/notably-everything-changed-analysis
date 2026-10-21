#!/usr/bin/env python3
"""Stage 1 of the historical calibration (task #5): build editing pairs from WildChat-1M under the frozen
rules of results/historical_calibration/PROTOCOL.md. Writes the pairs OUTSIDE the repo (they contain user
text) and only aggregate counts into results/historical_calibration/selection*.json.

Usage: python select_pairs.py <wildchat_dir> <pairs_out.jsonl> [--maxwords 200]
"""
import ast
import difflib
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
REPO = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
from features import LLM, WORD  # noqa: E402,F401  (ok_para's globals)


def load_funcs(path, names, ns):
    """Execute only the named top-level function definitions of a script, so definitions cannot drift."""
    tree = ast.parse(open(path).read())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    return ns


NS = load_funcs(os.path.join(SCRIPTS, "rewrite.py"), {"ok_para"}, {"re": re, "LLM": LLM})
ok_para = NS["ok_para"]

EDIT = re.compile(r"\b(polish|proofread|revise|rewrite|re-write|paraphrase|rephrase|improve|edit|refine|"
                  r"correct (the )?grammar|fix (the )?grammar|make (it|this|them)( sound)? (more )?"
                  r"(academic|formal|professional|fluent))", re.I)
ACAD_REQ = re.compile(r"\b(paper|manuscript|abstract|journal|conference|thesis|dissertation|research|scientific|academic)\b", re.I)
CITE = re.compile(r"\[\d+(,\s*\d+)*\]|\bet al\.|\([A-Z][A-Za-z-]+,? (19|20)\d\d[a-z]?\)")
# Deviation 1 (see PROTOCOL.md): the instruction must be short and the edit verb must govern a text object
EDIT2 = re.compile(r"\b(polish|proofread|revise|rewrite|re-write|paraphrase|rephrase|improve|edit|refine|correct|fix)\b"
                   r"[^.\n]{0,60}?\b(this|these|the following|following|my|it|below|above|text|paragraph|passage|"
                   r"abstract|sentences?|essay|introduction|section|writing|grammar)\b"
                   r"|\bmake (it|this|them)( sound)? (more )?(academic|formal|professional|fluent)", re.I)
B_GRAM = re.compile(r"\b(grammar|proofread|spelling|typos?)\b", re.I)
B_NOTGRAM = re.compile(r"\b(polish|improve|rewrite|re-write|paraphrase|rephrase)\b", re.I)
B_REW = re.compile(r"\b(rewrite|re-write|paraphrase|rephrase)\b", re.I)
B_ACAD = re.compile(r"\b(academic|paper|journal|conference|scientific|formal)\b", re.I)
PRE = re.compile(r"^[^\n]{0,250}:\s*$")
TRAIL = re.compile(r"^\s*(Changes|Explanation|Note|I made|I have|In this revision|Revisions)", re.I | re.M)


def words(t):
    return len(t.split())


def split_turn(text):
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if not blocks:
        return "", ""
    passage = max(blocks, key=words)
    instr = "\n".join(b for b in blocks if b is not passage)
    if words(passage) < 80 and len(blocks) > 1:
        # one block holding "instruction: passage": split off a leading instruction sentence/line
        m = re.match(r"^(.{0,300}?[:\n])\s*(.+)$", text.strip(), re.S)
        if m:
            instr, passage = m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()
    elif not instr:
        # Deviation 2: one block holding instruction + passage. Take the leading text up to a colon, or else
        # the first or the last sentence if it carries the edit request ("Polish this. <passage>").
        m = re.match(r"^(.{0,300}?:)\s*(.+)$", passage, re.S)
        if m:
            instr, passage = m.group(1), m.group(2).strip()
        else:
            sents = re.split(r"(?<=[.?!])\s+", passage.strip())
            if len(sents) > 2 and EDIT.search(sents[0]):
                instr, passage = sents[0], " ".join(sents[1:])
            elif len(sents) > 2 and EDIT.search(sents[-1]):
                instr, passage = sents[-1], " ".join(sents[:-1])
    return instr, re.sub(r"\s+", " ", passage).strip()


def clean_reply(r):
    r = r.strip()
    lines = r.split("\n")
    if lines and PRE.match(lines[0]) and words(lines[0]) <= 30:
        r = "\n".join(lines[1:]).strip()
    m = TRAIL.search(r)
    if m and m.start() > 0:
        r = r[: m.start()].strip()
    r = r.strip().strip('"“”').strip()
    return re.sub(r"\s+", " ", r)


def bucket(instr):
    if B_GRAM.search(instr) and not B_NOTGRAM.search(instr):
        return "grammar"
    if B_REW.search(instr):
        return "rewrite"
    if B_ACAD.search(instr):
        return "academic"
    return "polish"


def family(model):
    return "gpt-4" if model.startswith("gpt-4") else "gpt-3.5" if model.startswith("gpt-3.5") else model


MAXW = 200


def make_pair(user, assistant):
    """(instruction, passage, edited reply) for one user->assistant exchange, or the exclusion reason."""
    instr, passage = split_turn(user)
    if not EDIT.search(instr):
        return "x_no_edit_instruction"
    if words(instr) > 80 or not EDIT2.search(instr):
        return "x_instruction_not_an_edit_request"
    n = words(passage)
    if not (80 <= n <= MAXW):
        return "x_passage_length"
    # the paper's paragraph filter (ok_para also checks 80-200 words, so longer passages are probed on 150)
    probe = passage if n <= 200 else " ".join(passage.split()[:150]) + "."
    if not ok_para(probe if probe[-1] == "." else probe + ".") or not passage.rstrip().endswith("."):
        return "x_passage_quality"
    reply = clean_reply(assistant)
    if not (0.5 * n <= words(reply) <= 2.0 * n):
        return "x_reply_not_one_passage"
    # Deviation 1: an edit keeps the passage's content (word-sequence similarity >= 0.3)
    if difflib.SequenceMatcher(None, passage.lower().split(), reply.lower().split(), autojunk=False).ratio() < 0.3:
        return "x_reply_not_an_edit"
    return instr, passage, reply


def main():
    global MAXW
    wd, out = sys.argv[1], sys.argv[2]
    maxw = MAXW = int(sys.argv[sys.argv.index("--maxwords") + 1]) if "--maxwords" in sys.argv else 200
    why, seen, kept = Counter(), {}, []
    for f in sorted(glob.glob(os.path.join(wd, "train-*.parquet"))):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg, columns=["conversation_hash", "model", "timestamp", "conversation", "language"]).to_pylist():
                why["conversations"] += 1
                if r["language"] != "English":
                    why["x_language"] += 1
                    continue
                conv = r["conversation"]
                # Deviation 3: every user turn followed by an assistant turn is a candidate (edit requests are
                # often made mid-conversation); a conversation contributes at most one pair, its first valid one
                found = False
                fail = None
                for t in range(len(conv) - 1):
                    if conv[t]["role"] != "user" or conv[t + 1]["role"] != "assistant":
                        continue
                    res = make_pair(conv[t]["content"] or "", conv[t + 1]["content"] or "")
                    if isinstance(res, str):
                        fail = fail or res
                        if res not in ("x_no_edit_instruction",):
                            fail = res
                        continue
                    instr, passage, reply = res
                    found = True
                    break
                if not found:
                    why[fail or "x_no_first_exchange"] += 1
                    continue
                n = words(passage)
                key = hashlib.sha256(passage.lower().encode()).hexdigest()
                ts = r["timestamp"]
                if key in seen:
                    why["x_duplicate_passage"] += 1
                    if ts < seen[key]["ts"]:
                        seen[key].update(model=r["model"], family=family(r["model"]), ts=ts, reply=reply, instr=instr[:500],
                                         bucket=bucket(instr), academic=bool(ACAD_REQ.search(instr) or CITE.search(passage)))
                    continue
                rec = {"key": key, "hash": r["conversation_hash"], "model": r["model"], "family": family(r["model"]),
                       "ts": ts, "instr": instr[:500], "passage": passage, "reply": reply,
                       "bucket": bucket(instr),
                       "academic": bool(ACAD_REQ.search(instr) or CITE.search(passage))}
                seen[key] = rec
    for rec in seen.values():
        rec["quarter"] = f"{rec['ts'].year}Q{(rec['ts'].month - 1) // 3 + 1}"
        rec["ts"] = rec["ts"].isoformat()
        kept.append(rec)
    why["kept_pairs"] = len(kept)
    with open(out, "w") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")
    strata = Counter((r["family"], r["quarter"], r["bucket"], r["academic"]) for r in kept)
    summary = {"maxwords": maxw, "counts": dict(why),
               "by_family_academic": dict(Counter(f"{r['family']}|academic={r['academic']}" for r in kept)),
               "by_quarter_family": dict(Counter(f"{r['quarter']}|{r['family']}" for r in kept)),
               "by_bucket_academic": dict(Counter(f"{r['bucket']}|academic={r['academic']}" for r in kept)),
               "models": dict(Counter(r["model"] for r in kept))}
    name = "selection.json" if maxw == 200 else f"selection_max{maxw}.json"
    json.dump(summary, open(os.path.join(REPO, "results", "historical_calibration", name), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
