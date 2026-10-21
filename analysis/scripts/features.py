"""Per-paper features: syntax, rhetoric, structure. Pure Python; used by analyze.py."""
import re

WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

# Sentence-final participle riders: ", highlighting X" (interpretive: comments on meaning) vs
# ", enabling X" (functional: states a consequence). Lists fixed before looking at post-AI data,
# from the riders the earlier MLSys/OSDI scan found plus their common synonyms.
INTERP_CORE = ("highlighting underscoring reflecting showcasing emphasizing signaling signalling "
               "underlining marking paving reinforcing affirming attesting").split()
# Weak tier: also ordinary result-reporting verbs (", indicating that S stores ...").
INTERP_WEAK = "demonstrating illustrating revealing indicating suggesting confirming validating".split()
INTERP = INTERP_CORE + INTERP_WEAK
FUNC = ("enabling ensuring allowing resulting leading achieving yielding reducing improving making "
        "providing requiring causing preventing eliminating avoiding increasing decreasing "
        "producing creating forcing limiting incurring saving lowering raising").split()

HEDGE = re.compile(r"\b(may|might|could|possibly|potentially|likely|perhaps|seems?|appears?|presumably|arguably)\b", re.I)
# Token sets are disjoint across the index features: "Notably," counts only as an opener,
# "underscoring"/"showcasing" only as riders, "crucial(ly)" only as a style word.
BOOST = re.compile(r"\b(significantly|substantially|dramatically|remarkably|"
                   r"fundamentally|greatly|considerably|drastically|markedly|vastly)\b", re.I)
WE = re.compile(r"\b(we|our|us|ours)\b", re.I)
OPENERS = {
    "moreover": r"Moreover", "furthermore": r"Furthermore", "additionally": r"Additionally",
    "notably": r"Notably", "importantly": r"(?:More |Most )?[Ii]mportantly", "crucially": r"Crucially",
    "overall": r"Overall", "in_summary": r"In summary|To summarize|In short", "however": r"However",
    "specifically": r"Specifically", "to_this_end": r"To this end", "in_particular": r"In particular",
    "thus": r"Thus|Hence|Therefore", "for_example": r"For example|For instance",
    "intuitively": r"Intuitively", "unlike": r"Unlike", "while": r"While",
}
OPENER_RE = {k: re.compile(rf"^(?:{v})\b") for k, v in OPENERS.items()}
THIS_CLOSER = re.compile(r"^(?:This|These|Such|It|That)\s+(?:\w+\s+){0,3}?(shows|demonstrates|highlights|"
                         r"underscores|suggests|indicates|reveals|confirms|illustrates|showcases|"
                         r"emphasizes|validates|underlines)\b")
TRIAD = re.compile(r"\b([a-z]{3,}), ([a-z]{3,}),? and ([a-z]{3,})\b")
RUNIN = re.compile(r"^([A-Z][A-Za-z0-9/&' -]{1,40}?)(?: \d+)?[.:]\s+[A-Z(]")
RUNIN_INSIGHT = re.compile(r"^(?:Key )?(Insight|Idea|Takeaway|Observation|Finding|Lesson|Challenge|Opportunity|"
                           r"Implication|Remark|Intuition)s?(?: \d+| #\d+)?[.:]", re.I)
QUANT = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s?(?:%|×|x\b|X\b)")
NUM = re.compile(r"(?<![\w.\[-])\d+(?:\.\d+)?(?![\w\]])")
UPTO = re.compile(r"\bup to \d", re.I)
NOVEL = re.compile(r"\bnovel\b", re.I)
FIRST = re.compile(r"\b(?:the|is|are|be) first\b|\bfirst to\b|to (?:the best of )?our knowledge", re.I)
SOTA = re.compile(r"state[- ]of[- ]the[- ]art|\bSOTA\b", re.I)
LLM = re.compile(r"\bLLMs?\b|\bM?LLMs?\b|\bVLMs?\b|large language models?|vision[- ]language models?|\bGPT-?[34]|\bChatGPT\b|"
                 r"\bLlama|\bQwen|\bGemini\b|\bMistral\b|\bDeepSeek|\bClaude\b|\bfoundation models?\b", re.I)


# Function words for a topic-free style profile (homogenization test).
FW = ("the of and to a in is that for we this on with as by are be it which from an can our not or at "
      "these each such than also more has have only but when while however thus both all its their "
      "they there into over under between through without within among since because although though "
      "whereas whether if then so very much many most some any other another same different further "
      "first second finally instead rather particular particularly especially "
      "however, therefore hence indeed often typically usually generally may might could would should "
      "must will do does done here where how what why who whose").replace(",", "").split()
FW = list(dict.fromkeys(FW))
# Word lists fixed before looking at this corpus. FAMOUS: the ChatGPT words that rose in
# biomedical abstracts (Kobak et al. 2024; Liang et al. 2024). STYLE: words the earlier
# MLSys/OSDI 2021-vs-2025 scan (roast-paper calibration) found rising in systems papers.
FAMOUS = re.compile(r"\b(delv\w*|tapestr\w*|testament|realm|meticulous\w*|commendabl\w*|intricac\w*|"
                    r"showcas(?:e|es|ed)|unwavering|groundbreaking|noteworthy|invaluable|elucidat\w*)\b", re.I)
STYLE_LEX = re.compile(r"\b(seamless(?:ly)?|holistic(?:ally)?|underscor(?:e|es|ed)|intricate|"
                       r"pivotal|cornerstone|plays? (?:a )?(?:crucial|critical|pivotal|key|vital) role|"
                       r"crucial(?:ly)?|vital|paramount|landscape|nuanced|multifaceted|foster\w*|"
                       r"streamlin\w*|bolster\w*|harness\w*|unlock\w*|navigat\w*|"
                       r"comprehensive(?:ly)?|robust(?:ly|ness)?|enhanc\w*|leverag\w*)\b", re.I)


def mattr(tokens, w=500):
    """Moving-average type-token ratio (length-independent lexical diversity)."""
    if len(tokens) < w:
        return len(set(tokens)) / max(len(tokens), 1)
    from collections import Counter
    c = Counter(tokens[:w])
    tot = len(c)
    for i in range(w, len(tokens)):
        c[tokens[i]] += 1
        c[tokens[i - w]] -= 1
        if c[tokens[i - w]] == 0:
            del c[tokens[i - w]]
        tot += len(c)
    return tot / (len(tokens) - w + 1) / w


def sentences(text):
    return [s for s in SENT_SPLIT.split(text) if len(s.split()) >= 4]


def intro_opener(s):
    s = s.strip()
    if re.match(r"(?:The )?(?:recent |rapid )?(?:advent|emergence|rise|success|advances?|progress|proliferation|popularity|growth)\b", s, re.I) \
            or re.match(r"(?:With|Due to|Driven by|Thanks to) (?:the )?(?:rapid|recent|increasing|growing|remarkable|unprecedented|explosive)", s, re.I):
        return "rise-of-X"
    if re.match(r"(?:In recent years|Recently|Over the (?:past|last)|In the (?:past|last)|Nowadays|Today|Modern)\b", s, re.I):
        return "recently"
    if LLM.match(s) or re.match(r"(?:Large|Multimodal|Vision)[- ](?:language|Language)", s):
        return "llm-subject"
    if re.search(r"\b(?:has|have) (?:become|emerged|gained|attracted|revolutionized|transformed|shown|achieved|demonstrated)\b", s[:160]):
        return "has-become"
    if re.match(r"(?:Many|Most|A (?:wide|large|growing|key)|Numerous|An increasing)\b", s):
        return "many-X"
    if re.match(r"[A-Z][\w-]*(?: [\w-]+){0,5} (?:is|are) (?:a|an|the|one)\b", s):
        return "definition"
    return "other"


def ing_riders(sents):
    interp = core = func = other = 0
    hits = []
    for s in sents:
        for m in re.finditer(r",\s+(?:thereby |thus |further |ultimately |effectively )?([a-z]+ing)\b", s):
            v = m.group(1)
            # Only count riders after a main clause (not inside lists or at sentence start).
            if m.start() < 25:
                continue
            if v in INTERP:
                interp += 1
                core += v in INTERP_CORE
                hits.append(("interp", v))
            elif v in FUNC:
                func += 1
            elif v not in ("including", "using", "following", "according", "during", "being",
                           "having", "regarding", "string", "nothing", "something", "anything",
                           "everything", "padding", "bring", "thing", "king", "ring", "spring"):
                other += 1
    return interp, core, func, other, hits


POLICY_HEAD = re.compile(r"limitation|ethic|broader|impact statement|impact", re.I)
POLICY_START = re.compile(r"^(?:Limitations?|Ethics(?: Statement)?|Ethical Considerations|Broader Impacts?|Impact Statement)\b[.:]?")
_norm = lambda x: re.sub(r"[^a-z]", "", x.lower())


def drop_policy(rec):
    """Remove paragraphs from Limitations / Ethics / Impact sections, which venue rules made
    mandatory in the post period and which have their own register."""
    blob = _norm(" ".join(t for h, _, t in rec.get("segments", []) if POLICY_HEAD.search(h) and t))
    keep = []
    for p in rec["paras"]:
        if POLICY_START.match(p):
            continue
        key = _norm(p)[:60]
        if blob and len(key) >= 40 and key in blob:
            continue
        keep.append(p)
    return keep


def paper_features(rec):
    paras = drop_policy(rec)
    text = "\n".join(paras)
    words = len(WORD.findall(text))
    if words < 1500:
        return None
    k = 1000.0 / words
    sents = [s for p in paras for s in sentences(p)]
    # Run-on "sentences" over 80 words are extraction artifacts (equations, lists): skip for length stats.
    lens = [len(s.split()) for s in sents if len(s.split()) <= 80]
    mean = sum(lens) / len(lens)
    sd = (sum((x - mean) ** 2 for x in lens) / len(lens)) ** 0.5
    interp, core, func, other, _ = ing_riders(sents)
    f = {
        "words": words, "pages": rec.get("pages") or 0,
        "sent_len": mean, "sent_sd": sd, "sent_cv": sd / mean,
        "para_len": words / len(paras),
        "emdash_1k": text.count("—") * k,
        "endash_spaced_1k": text.count(" – ") * k,
        "ing_interp_1k": interp * k, "ing_interp_core_1k": core * k, "ing_func_1k": func * k, "ing_other_1k": other * k,
        "rather_than_1k": len(re.findall(r"\brather than\b", text, re.I)) * k,
        "this_closer_1k": sum(bool(THIS_CLOSER.match(s)) for s in sents) * k,
        "triad_1k": len(TRIAD.findall(text)) * k,
        "hedge_1k": len(HEDGE.findall(text)) * k,
        "boost_1k": len(BOOST.findall(text)) * k,
        "we_1k": len(WE.findall(text)) * k,
        "semicolon_1k": text.count(";") * k,
        "question_1k": text.count("?") * k,
        "colon_1k": len(re.findall(r"[a-z]: [a-z]", text)) * k,
        "not_only_1k": len(re.findall(r"\bnot (?:only|just|merely|simply)\b", text, re.I)) * k,
        "quant_1k": len(QUANT.findall(text)) * k,
        "num_per_para": sum(len(NUM.findall(p)) for p in paras) / len(paras),
        "upto_1k": len(UPTO.findall(text)) * k,
        "novel_1k": len(NOVEL.findall(text)) * k,
        "first_claim_1k": len(FIRST.findall(text)) * k,
        "sota_1k": len(SOTA.findall(text)) * k,
        "runin_share": sum(bool(RUNIN.match(p)) for p in paras) / len(paras),
        "runin_insight_1k": sum(bool(RUNIN_INSIGHT.match(p)) for p in paras) * k,
        "llm_1k": len(LLM.findall(text)) * k,
        "famous_ai_words_1k": len(FAMOUS.findall(text)) * k,
        "style_lex_1k": len(STYLE_LEX.findall(text)) * k,
    }
    toks = [w.lower() for w in WORD.findall(text)]
    f["mattr"] = mattr(toks)
    from collections import Counter
    tc = Counter(toks)
    f["fw"] = [1000.0 * tc[w] / len(toks) for w in FW]
    for name, rx in OPENER_RE.items():
        f["op_" + name + "_1k"] = sum(bool(rx.match(s)) for s in sents) * k
    # Structure.
    segs = [(h, n, t) for h, n, t in rec["segments"] if h != "(front)"]
    real = [s for s in segs if not re.search(r"acknowledg", s[0]) and s[1] >= 30]
    f["n_sections"] = len(real)
    get = lambda rx: [(h, n, t) for h, n, t in segs if re.search(rx, h)]
    intro = get(r"^introduction")
    intro_txt = intro[0][2] if intro else ""
    f["intro_words"] = intro[0][1] if intro else 0
    f["related_words"] = sum(n for _, n, _ in get(r"related"))
    f["has_related"] = int(bool(get(r"related")))
    rel_idx = next((i for i, s in enumerate(real) if "related" in s[0]), None)
    f["related_early"] = int(rel_idx is not None and rel_idx <= 2)
    f["has_limitations"] = int(bool(get(r"limitation")) or bool(re.search(r"\n(?:Limitations?|LIMITATIONS?)[.:]\s", text)))
    f["limitations_words"] = sum(n for _, n, _ in get(r"limitation"))
    f["has_discussion"] = int(bool(get(r"discussion")))
    f["discussion_words"] = sum(n for _, n, _ in get(r"discussion"))
    f["has_ethics_impact"] = int(bool(get(r"ethic|broader|impact statement")))
    f["conclusion_words"] = sum(n for _, n, _ in get(r"conclusion|summary"))
    f["has_future_work"] = int(bool(re.search(r"future work|future direction", text, re.I)))
    contrib = re.search(r"contribution", intro_txt, re.I)
    enum = intro_txt.count("•") >= 2 or len(re.findall(r"\((?:[1-5]|i{1,3}|iv|v)\)", intro_txt)) >= 2 \
        or len(re.findall(r"(?:^|\n)\s*(?:[1-5]\.|[–-]) [A-Z]", intro_txt)) >= 2
    f["contrib_list"] = int(bool(contrib and enum))
    f["contrib_bullets"] = int(bool(contrib) and intro_txt.count("•") >= 2)
    f["n_figs"], f["n_tabs"], f["n_refs"] = rec["n_figs"], rec["n_tabs"], rec["n_refs"]
    f["github"] = int(rec["github"] > 0)
    f["artifact"] = int(rec["artifact"])
    f["has_appendix"] = int(rec["has_appendix"])
    f["abstract_words"] = len(rec["abstract"].split())
    f["abstract_quant"] = len(QUANT.findall(rec["abstract"]))
    f["abstract_has_quant"] = int(f["abstract_quant"] > 0)
    # Rhetoric: how the intro opens (first sentence after the heading).
    first = sentences(re.sub(r"\s+", " ", intro_txt))[:1]
    f["intro_open"] = intro_opener(first[0]) if first else "none"
    f["intro_first"] = first[0][:300] if first else ""
    f["llm_topic"] = int(len(LLM.findall(text)) >= 5)
    return f
