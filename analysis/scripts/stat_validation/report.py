#!/usr/bin/env python3
"""Render aggregate result synopsis after validate.py completes."""
import hashlib, json
from pathlib import Path
import pandas as pd
base=Path(__file__).resolve().parents[2]/'results/stat_validation'
manifest=json.loads((base/'manifest.json').read_text())
frozen=base.parent/'corpus_validation/frozen_inputs.json'
manifest['style_complete_case_exclusions']=0
manifest['original_source_sha256']=json.loads(frozen.read_text())['source_hashes']
manifest['frozen_provenance_sha256']=hashlib.sha256(frozen.read_bytes()).hexdigest()
manifest['audit_script_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
(base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
summary={}
for cohort in ['legacy_mapping','corrected_vldb2016','legacy_mapping_no_iclr','corrected_vldb2016_no_iclr','corrected_no_late_arxiv']:
    cl=pd.read_csv(base/f'{cohort}_classifier_summary.csv')
    shift=pd.read_csv(base/f'{cohort}_shift.csv')
    cp=pd.read_csv(base/f'{cohort}_changepoint.csv')
    loo=pd.read_csv(base/f'{cohort}_leave_one_venue_out.csv')
    h=pd.read_csv(base/f'{cohort}_historical_scores.csv')
    h=h[(h.feature_set=='style33')&(h.normalization=='venue')&(h.year<=2022)]
    summary[cohort]={'classifier_style33':cl[cl.feature_set=='style33'].to_dict('records'),'index9_shifts':shift[shift.feature_set=='index9'].to_dict('records'),'index9_changepoints':cp.fillna('').to_dict('records'),'trend_loo_ranges':loo[(loo.feature_set=='index9')&(loo.trend_adjusted.astype(str).str.lower()=='true')].groupby(['scaling','weight']).estimate.agg(['min','max']).reset_index().to_dict('records'),'historical_pre2023_share_score_gt_half':float((h.share_score_gt_half*h.n).sum()/h.n.sum())}
(base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Strict statistical validation','',
'All results describe historical era separation and changes in the extracted feature corpus. They do not identify AI use, estimate AI authorship prevalence, or establish a causal effect of AI. Analyses are exploratory robustness checks, not preregistered confirmation.','',
'## Reproduction','',
'Run from `output/before-after-ai` with a scientific Python environment:','',
'```sh',
'python scripts/stat_validation/check.py',
'python scripts/stat_validation/validate.py --features results/corpus_validation/features_numeric.jsonl.gz --out results/stat_validation --bootstrap 500',
'python scripts/stat_validation/report.py',
'```','',
'Alternatively use `--source /path/to/frozen/v3-20261019` for the private original features.jsonl. The numerical base features alone determine all analyses; newfeatures.jsonl and all_features.csv are not merged. All retained base STYLE/INDEX9 observations are finite (zero complete-case exclusions); no missing controls are imputed. The source manifest records file hashes, definition hashes, exact sample counts, Python/package versions, seed 20261019 and 500 bootstrap replicates. It reads STYLE (33) and INDEX9 (9) literals from original scripts without modifying or executing legacy analysis. Source hashes from private cache, when supplied, are provenance only. No model calls or downloads occur.','',
'## Cohorts and exclusions','',
'The original venue/era mapping yields 9,023 papers in 98 cells. It excludes 371 papers from the four starred/cross-venue-baseline cells (COLM/EuroSys) and 78 VLDB2016 papers. The corrected cohort includes those 78 VLDB2016 controls: 9,101 papers in 99 cells. Starred venues are excluded throughout. `cohort_counts.csv` records counts by cohort/venue/year/era; year is derived from the corpus name. NeurIPS2024 retains its original post label and MLSys2019 its original pre label.','',
'The no-ICLR cohorts contain 8,147 papers (legacy) and 8,225 (corrected), each in 89/90 venue-year cells respectively; the corrected date-exclusion cohort contains 9,083 papers in 99 cells. Proceedings-only sensitivity separately removes ICLR from both cohorts; it is a source-format robustness check, not a revision of the main cohort. The fifth cohort removes only the 18 active pre-2023 papers whose cached arXiv version stamps occur after 2022-11-30, matched using the corpus audit hashed key. Venue years are never relabeled from version dates. This date sensitivity detects known later versions; it cannot rule out undated revisions or source differences elsewhere.','',
'## Classifier contracts','',
'Logistic regression uses C=0.3, max_iter=2000. The same StratifiedKFold(5, shuffle=True, random_state=0) splits compare preprocessing on pre/post papers only. The legacy arm estimates each venue baseline mean/SD from all pre rows before splitting; this lets held-out pre observations influence preprocessing. Strict arms estimate baseline mean and SD solely from training pre rows, then apply fixed clipping [-8,8] to both training and held-out data. Era is known only in training and selects the historical baseline; no held-out label selects its transform. A pooled-SD sensitivity estimates SD from all training pre/post rows but continues centering on training pre means. Baseline zero SD falls back to 1, matching legacy classifier behavior.','',
'Fold-local venue scaling evaluates familiar venues with historical examples. Unseen-venue LOVO learns one pooled baseline mean/SD from other venues\' training pre papers and uses no target-venue observations for scaling, fitting or clipping. Legacy LOVO intentionally calibrates on all target pre papers, including evaluated pre papers, and is labeled accordingly. Disjoint target-baseline LOVO randomly selects half of target pre papers (seed fixed per venue), excludes them from evaluation, and uses them solely to estimate the target baseline transform; the classifier trains only on other venues. It measures a different information contract and a different evaluation sample from fully unseen-venue LOVO. These arms are not interchangeable claims of venue generalization.','',
'`historical_scores.csv` fits on pre/post rows and scores only control/trend/series years. It reports mean era score and share >0.5 under venue or pooled historical normalization. Earlier-year scores are descriptive negative-control diagnostics, not calibrated probabilities of AI use. The classifier outputs fold/venue AUCs, sample counts, mean AUC and sample SD (ddof=1); fold SD is not a confidence interval. Paper-level CV cannot establish author-disjoint or source-disjoint performance; author/family identifiers are unavailable here.','',
'## Index, trend, weighting and uncertainty','',
'INDEX9 averages nine features centered on venue pre means, divided either by venue pre+post SD (legacy descriptive index) or pre-only SD. Feature denominator floor is .001, matching legacy index. A third scaling takes the legacy composite and then restandardizes the composite itself by venue pre SD: this reproduces legacy stats.py part_b units, which differ from the raw composite. Removing the two lexical lists (INDEX7), and additionally em-dash (INDEX6), audits feature choice. These are sensitivity sets selected during review.','',
'The unadjusted contrast fits venue intercepts plus a common post indicator on original pre/post samples. The trend-adjusted contrast fits venue intercepts, venue-specific linear year slopes, and a common step from 2024 onward; it uses years <=2022 or >=2024 and excludes mixed year 2023. Full year <=2022 sampling, including earlier controls, determines the extrapolation. Paper weighting exactly reproduces paper-level OLS coefficients using cell means and counts; cell weighting gives every venue-year equal weight. These contrasts answer different weighting questions.','',
'Each confidence interval resamples whole venue trajectories, preserving arbitrary serial dependence within a venue, with 500 deterministic replicates and percentile 95% bounds. The linear coefficient bootstrap uses an algebraically equivalent within-venue residualized cross-product implementation. The check script verifies it against explicit resampled regressions. This is a sensitivity to venue composition under an exchangeability assumption: the 10 (or 9) convenience-sampled venues do not justify broad population confidence. Leave-one-venue-out estimates expose concentration. The venue bootstrap assumes different venue trajectories are exchangeable and does not model shared global-year shocks across venues. No paper-level independence assumption is used for it; no cluster-asymptotic p-values are reported.','',
'## Changepoint','',
'Fit equal-weight venue-year means through 2025 to venue intercepts + a common linear trend, versus the same model plus a step at candidate year tau. BIC uses number of cells and design rank, not number of papers. The unrestricted legacy candidate set is 2021–2025; the constrained sensitivity is 2021–2024, requiring at least two distinct observed years globally on each side of the break and at least one year per side in each venue. NeurIPS has only one post-2024 year; this is not a two-years-per-venue condition. Candidate years and both windows are explicit in output. All ten series and ACL/ICML-only are reported separately.','',
'Whole-venue bootstrap retains full time trajectories and captures composition sensitivity. The separate paper-within-fixed-cell bootstrap keeps venue/year cells and fitted scale denominators fixed; it measures conditional sampling variability of cell means, not population venue uncertainty. ACL/ICML has only two independent trajectories: its venue bootstrap is correspondingly coarse. Break-year bootstrap frequencies are selection stability over a prespecified candidate grid, not posterior probabilities or confidence that an event caused a break. Leave-one-venue-out BIC/breaks are also reported.','',
'## Results','',
'| Cohort | 33-feature legacy CV AUC | strict fold venue-pre AUC | fully unseen-venue LOVO mean AUC | trend shift (legacy composite pre SD; paper weighted) | venue-bootstrap 95% interval |',
'|---|---:|---:|---:|---:|---|']
for cohort,result in summary.items():
    cl=pd.DataFrame(result['classifier_style33']).set_index('scheme')
    sh=pd.DataFrame(result['index9_shifts']);r=sh[(sh.scaling=='pooled_sd_then_composite_pre_sd')&(sh.weight=='paper')&(sh.trend_adjusted)].iloc[0]
    lines.append(f"| {cohort} | {cl.loc['legacy_full_data_venue_pre','mean_auc']:.6f} | {cl.loc['strict_fold_venue_pre','mean_auc']:.6f} | {cl.loc['strict_lovo_unseen_venue_pooled_pre','mean_auc']:.6f} | {r.estimate:.6f} | [{r.venue_boot_lo:.6f}, {r.venue_boot_hi:.6f}] |")
lines+=['','Full exact values and all feature/scaling/weight sensitivities are in `summary.json` and the CSV files. AUC remains high after fold-local scaling in this corpus; that persistence does not remove source/version, topic, extraction, or author-composition confounding. The sign of a trend residual is conditional on linear extrapolation, chosen feature definitions and cohort selection.','']
(base/'README.md').write_text('\n'.join(lines))
print('Wrote summary.json and README.md')
