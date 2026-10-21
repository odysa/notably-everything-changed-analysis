#!/usr/bin/env python3
"""Render the CSVs from analyze.py into Markdown tables for the report.

Usage: python3 -I tables.py <results_dir>   -> writes <results_dir>/tables.md
"""
import csv, math, os, sys

R = sys.argv[1] if __name__ == "__main__" else None
rd = lambda name: list(csv.DictReader(open(os.path.join(R, name))))
fl = lambda x: float(x) if x not in ("", "nan", None) else float("nan")

LABEL = {
    "ing_interp_core_1k": "interpretive -ing rider, core (/1k)", "ing_interp_1k": "interpretive -ing rider, all (/1k)",
    "ing_func_1k": "functional -ing rider (/1k)", "ing_other_1k": "other -ing rider (/1k)",
    "emdash_1k": "em-dash (/1k)", "rather_than_1k": "'rather than' (/1k)", "this_closer_1k": "'This shows/highlights' sentence (/1k)",
    "triad_1k": "'X, Y, and Z' triad (/1k)", "hedge_1k": "hedges (/1k)", "boost_1k": "boosters (/1k)",
    "we_1k": "we/our/us (/1k)", "semicolon_1k": "semicolons (/1k)", "question_1k": "questions (/1k)",
    "not_only_1k": "'not only/just/merely' (/1k)", "quant_1k": "quantified claims N%/N× (/1k)",
    "num_per_para": "numbers per paragraph", "upto_1k": "'up to N' (/1k)", "novel_1k": "'novel' (/1k)",
    "first_claim_1k": "'first'/'to our knowledge' (/1k)", "sota_1k": "'state of the art' (/1k)",
    "runin_share": "share of paragraphs with run-in label", "runin_insight_1k": "'Insight:/Takeaway:/Observation:' labels (/1k)",
    "llm_1k": "LLM mentions (/1k)", "sent_len": "mean sentence length (words)", "sent_sd": "sentence length SD",
    "sent_cv": "sentence length CV", "para_len": "mean paragraph length (words)", "words": "body prose words",
    "pages": "PDF pages", "n_sections": "top-level sections", "intro_words": "intro words",
    "related_words": "related-work words", "has_related": "has related-work section", "related_early": "related work in first 3 sections",
    "has_limitations": "has limitations section/paragraph", "limitations_words": "limitations words",
    "has_discussion": "has discussion section", "discussion_words": "discussion words",
    "has_ethics_impact": "has ethics/impact statement", "conclusion_words": "conclusion words",
    "has_future_work": "mentions future work", "contrib_list": "intro has enumerated contribution list",
    "contrib_bullets": "intro has bulleted contribution list", "n_figs": "figures", "n_tabs": "tables",
    "n_refs": "references", "github": "links GitHub", "artifact": "mentions artifact evaluation",
    "has_appendix": "has appendix", "abstract_words": "abstract words", "abstract_has_quant": "abstract has N%/N× claim",
    "abstract_quant": "N%/N× in abstract", "style_lex_1k": "selected style words (/1k)",
    "famous_ai_words_1k": "famous ChatGPT words: delve, tapestry... (/1k)", "mattr": "lexical diversity (MATTR-500)", "colon_1k": "mid-sentence colons (/1k)", "endash_spaced_1k": "spaced en-dash (/1k)",
}
OP = {"op_" + k + "_1k": f"sentence opener '{v}' (/1k)" for k, v in {
    "moreover": "Moreover", "furthermore": "Furthermore", "additionally": "Additionally", "notably": "Notably",
    "importantly": "Importantly", "crucially": "Crucially", "overall": "Overall", "in_summary": "In summary",
    "however": "However", "specifically": "Specifically", "to_this_end": "To this end",
    "in_particular": "In particular", "thus": "Thus/Hence/Therefore", "for_example": "For example",
    "intuitively": "Intuitively", "unlike": "Unlike", "while": "While"}.items()}
LABEL.update(OP)
lab = lambda f: LABEL.get(f, f)


def fmt(x, nd=2):
    x = fl(x)
    if math.isnan(x):
        return "–"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.{nd}f}"




def main():
    out = []
    pool = rd("pooled.csv")
    out.append("## Pooled ranking (8 venues with a pre-AI year; COLM* and EuroSys* excluded)\n")
    out.append("| feature | pre mean | post mean | pre→post log ratio | mean AUC | AUC non-LLM papers | venues ↑/↓ | 2016→2020 AUC | rate/yr pre | rate/yr post | break share (ACL+ICML) | attribution |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in pool:
        out.append(f"| {lab(p['feature'])} | {fmt(p['pre_mean'])} | {fmt(p['post_mean'])} | {fmt(p['mean_log_ratio'])} | "
                   f"{fmt(p['mean_auc'])} | {fmt(p['mean_auc_nonllm'])} | {p['venues_up']}/{p['venues_down']} | "
                   f"{fmt(p['trend_mean_auc'])} | {fmt(p['rate_pre'], 3)} | {fmt(p['rate_post'], 3)} | {fmt(p['break_share'])} | {p['attribution']} |")

    pv = rd("per_venue.csv")
    KEY = ["words", "pages", "sent_len", "sent_sd", "para_len", "ing_interp_core_1k", "ing_func_1k", "emdash_1k",
           "boost_1k", "op_additionally_1k", "op_notably_1k", "this_closer_1k", "rather_than_1k", "triad_1k", "hedge_1k",
           "we_1k", "semicolon_1k", "quant_1k", "novel_1k", "contrib_list", "contrib_bullets", "has_limitations",
           "n_figs", "n_tabs", "n_refs", "related_words", "github", "llm_1k"]
    venues = []
    for r in pv:
        if r["venue"] not in venues:
            venues.append(r["venue"])
    for v in venues:
        rs = {r["feature"]: r for r in pv if r["venue"] == v}
        any_r = next(iter(rs.values()))
        out.append(f"\n### {v}  (n: 2016={any_r['n_ctrl']}, pre={any_r['n_pre']}, post={any_r['n_post']}, post non-LLM={any_r['n_post_nonllm']})\n")
        out.append("| feature | 2016 | pre | post | post non-LLM | share pre→post | AUC | AUC 2016→pre |")
        out.append("|---|---|---|---|---|---|---|---|")
        for f in KEY:
            r = rs.get(f)
            if not r:
                continue
            out.append(f"| {lab(f)} | {fmt(r['ctrl_mean']) if int(r['n_ctrl']) else '–'} | {fmt(r['pre_mean'])} | {fmt(r['post_mean'])} | "
                       f"{fmt(r['post_nonllm_mean'])} | {fmt(fl(r['pre_share']) * 100, 0)}%→{fmt(fl(r['post_share']) * 100, 0)}% | "
                       f"{fmt(r['auc'])} | {fmt(r['trend_auc'])} |")

    ser = rd("series.csv")
    SER = ["ing_interp_core_1k", "ing_func_1k", "emdash_1k", "boost_1k", "op_additionally_1k", "op_notably_1k",
           "this_closer_1k", "para_len", "sent_sd", "has_limitations", "contrib_bullets", "llm_1k", "words"]
    out.append("\n## Yearly series (ACL, ICML): where does the change happen?\n")
    out.append("| venue | feature | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for f in SER:
        for r in ser:
            if r["feature"] == f:
                out.append(f"| {r['venue']} | {lab(f)} | " + " | ".join(fmt(r[str(y)]) for y in range(2020, 2026)) + " |")

    op = rd("intro_openers.csv")
    out.append("\n## How introductions open (share of papers)\n")
    cats = ["rise-of-X", "recently", "llm-subject", "has-become", "many-X", "definition", "other"]
    out.append("| venue | era | n | " + " | ".join(cats) + " |")
    out.append("|---|---|---|" + "---|" * len(cats))
    for r in op:
        out.append(f"| {r['venue']} | {r['era']} | {r['n']} | " + " | ".join(f"{fl(r[c]) * 100:.0f}%" for c in cats) + " |")

    ts = rd("topic_share.csv")
    out.append("\n## Topic shift: share of papers that are LLM-centred (≥5 LLM mentions)\n")
    out.append("| corpus | n | LLM-centred |")
    out.append("|---|---|---|")
    for r in ts:
        out.append(f"| {r['corpus']} | {r['n']} | {fl(r['llm_topic_share']) * 100:.0f}% |")

    vc = rd("vocab.csv")
    def vtab(title, rows):
        out.append(f"\n## {title}\n")
        out.append("| term | pre df | post df | log ratio | log ratio non-LLM papers | venues ↑ | 2016→2020 log ratio |")
        out.append("|---|---|---|---|---|---|---|")
        for r in rows:
            out.append(f"| {r['term']} | {fl(r['pre_df']) * 100:.1f}% | {fl(r['post_df']) * 100:.1f}% | {fmt(r['mean_lr'])} | "
                       f"{fmt(r['mean_lr_nonllm'])} | {r['venues_up']}/{r['venues']} | {fmt(r['trend_lr'])} |")
    common = [r for r in vc if fl(r["post_df"]) >= 0.08 and int(r["venues"]) >= 6]
    vtab("Top 80 rising terms (post df ≥ 8%)", sorted(common, key=lambda r: -fl(r["mean_lr"]))[:80])
    robust = [r for r in common if fl(r["mean_lr_nonllm"]) > 0.5 and int(r["venues_up"]) >= 6]
    vtab("Rising terms that also rise in non-LLM papers and in ≥6 venues", sorted(robust, key=lambda r: -fl(r["mean_lr_nonllm"]))[:80])
    fall = [r for r in vc if fl(r["pre_df"]) >= 0.08 and int(r["venues"]) >= 6]
    vtab("Top 40 falling terms (pre df ≥ 8%)", sorted(fall, key=lambda r: fl(r["mean_lr"]))[:40])
    open(os.path.join(R, "tables.md"), "w").write("\n".join(out) + "\n")
    print("wrote", os.path.join(R, "tables.md"))


if __name__ == "__main__":
    main()
