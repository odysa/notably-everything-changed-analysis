#!/usr/bin/env python3
"""Lead's independent count/point-estimate and blinded-packet contract checks.

Does not establish annotation accuracy or reproduce PDFs/bootstrap intervals.
"""
import argparse, ast, collections, csv, gzip, hashlib, json, math, pathlib, statistics
ROOT = pathlib.Path(__file__).resolve().parents[2]
def read(p): return json.loads(p.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p): return [json.loads(x) for x in p.read_text().splitlines()]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--private-annotations', type=pathlib.Path)
    args = parser.parse_args()
    ext = ROOT/'results/review_round2/external'
    rows = lines(ext/'features_numeric.jsonl')
    sources = list(csv.DictReader((ext/'sources.csv').open()))
    statuses = read(ext/'retrieval.json')
    summary = read(ext/'summary.json')
    assert len(rows) == len(sources) == len(statuses) == 80
    assert len({r['paper_hash'] for r in rows}) == len({r['DOI'] for r in sources}) == 80
    assert {r['paper_hash'] for r in rows} == {r['paper_hash'] for r in sources}
    assert collections.Counter(r['year'] for r in rows) == {2019:20,2020:20,2024:20,2025:20}
    assert all(r['status'] == 'included' for r in statuses)
    allowed = {'paper_hash','year','words','ing_func_1k','ing_interp_core_1k','op_additionally_1k','op_notably_1k'}
    assert all(set(r) == allowed and r['words'] >= 1500 for r in rows)
    metrics = ('ing_func_1k','ing_interp_core_1k','addnot_1k')
    point = {}
    for key in metrics:
        value = lambda r: r['op_additionally_1k']+r['op_notably_1k'] if key == 'addnot_1k' else r[key]
        delta = statistics.fmean(value(r) for r in rows if r['year'] > 2023) - statistics.fmean(value(r) for r in rows if r['year'] < 2023)
        assert math.isclose(delta, summary['results'][key]['post_minus_pre_1k'], abs_tol=1e-12)
        point[key] = delta
    assert summary['confirmation'] is False
    for name, expected in summary['hashes'].items():
        assert digest(ROOT/name) == expected, name
    out = {'status':'passed', 'scope':'independent stdlib counts and point estimates; optional frozen annotation contracts; not human validation', 'plos_n':80, 'plos_independent_point_estimates':point, 'plos_joint_confirmation':False, 'input_hashes':{str(p.relative_to(ROOT)):digest(p) for p in [ext/'features_numeric.jsonl',ext/'summary.json',ext/'sources.csv']}}
    if args.private_annotations:
        private = args.private_annotations
        packet = {r['item_id']:r['sentence'] for r in lines(private/'packet.jsonl')}
        mapping = {r['item_id']:r for r in lines(private/'mapping.jsonl')}
        assert len(packet) == len(mapping) == 411
        assert all(set(r)=={'item_id','sentence'} for r in lines(private/'packet.jsonl'))
        groups = collections.Counter(r['stratum'] for r in mapping.values())
        assert groups == {'random':320,'enriched':91}
        for who in ('A','B'):
            annotations = {r['item_id']:r for r in lines(private/f'annotations_{who}.jsonl')}
            assert len(annotations) == 411 and set(annotations) == set(packet)
            for item, row in annotations.items():
                for clause in row['clauses']:
                    assert 0 <= clause['start'] <= clause['verb_start'] < clause['verb_end'] <= clause['end'] <= len(packet[item])
        receipt = read(ROOT/'results/review_round2/annotation/descriptive_audit.json')
        for name, expected in receipt['input_sha256'].items(): assert digest(private/name) == expected
        out['annotation_packet_n'] = len(packet)
        out['annotation_strata'] = dict(groups)
        out['annotation_frozen_input_hashes'] = receipt['input_sha256']
    repro = ROOT/'results/review_round2/reproduction'
    if (repro/'r1_full.json').exists():
        full = read(repro/'r1_full.json')
        inv, run = full['inventory'], full['reextraction']
        assert inv['list_entries'] == 9626 and sum(inv['fate'].values()) == 9626
        assert inv['fate'] == {'acl2020_short_paper_filter':75,'feature_row':9472,'no_pdf':4,'no_text_record':3,'under_1500_words':72}
        assert all(full['cache_hashes_match_frozen'].values())
        assert run['papers'] == run['text_record']['identical'] == run['features_bit_identical_to_cache'] == run['features_bit_identical_to_public_snapshot'] == 9472
        assert not run['mismatched_papers']
        pdfs = list(csv.DictReader((repro/'pdf_sha256.csv').open()))
        cached = [r for r in pdfs if r['pdf_sha256']]
        assert len(pdfs) == 9626 and len(cached) == 9622
        network = read(repro/'r2_redownload.json')['rows']
        sampled = [r for r in network if r['kind'] == 'sample']
        assert len(sampled) == 23 and all(r['bytes_identical'] for r in sampled)
        out['reproduction_receipt_audit'] = {'scope':'receipt and inventory consistency; lead did not repeat full extraction','analysed_text_and_features_exact_n':9472,'source_redownload_sample_identical_n':23,'cached_pdf_count':9622,'cached_pdf_bytes':sum(int(r['pdf_bytes']) for r in cached),'inventory_fates':inv['fate']}
        out['input_hashes'].update({str(q.relative_to(ROOT)):digest(q) for q in [repro/'r1_full.json',repro/'r2_redownload.json',repro/'pdf_sha256.csv']})
    catalog_path = ROOT/'results/review_round2/expansion/catalog_entries.csv'
    if catalog_path.exists():
        catalog = list(csv.DictReader(catalog_path.open()))
        expected = {'cvpr2020':1316,'cvpr2025':2721,'icml2020':934,'icml2025':3180,'neurips2020':1748,'neurips2025':5286}
        assert len(catalog) == 16472
        assert len({(r['cell'],r['source_id']) for r in catalog}) == len(catalog)
        candidate_counts = collections.Counter(r['cell'] for r in catalog if r['expansion_catalog_eligible']=='1')
        assert candidate_counts == expected
        assert sum(r['in_original_manifest']=='1' for r in catalog) == 750
        assert all(not (r['in_original_manifest']=='1' and r['expansion_catalog_eligible']=='1') for r in catalog)
        assert all(n >= 200 for n in candidate_counts.values())
        out['expansion_catalog_audit'] = {'catalog_entries':len(catalog),'original_manifest_members':750,'source_disjoint_catalog_candidates':dict(candidate_counts),'proposed_selected_n':1200,'pdfs_or_outcomes_retrieved':False}
        out['input_hashes'][str(catalog_path.relative_to(ROOT))] = digest(catalog_path)
    recovery_path = ROOT/'results/review_round2/recovery/sensitivity.json'
    if recovery_path.exists():
        recovered = read(recovery_path)
        manifest = ROOT/'results/review_round2/recovery/manifest.json'
        assert recovered['manifest_sha256'] == digest(manifest)
        additions = recovered['items']
        assert len(additions) == recovered['n_admitted'] == 7
        assert all(r['status']=='admitted' and r['words']>=1500 for r in additions)
        assert sum(r['role'].startswith('post:') for r in additions)==6
        assert [r['corpus'] for r in additions if not r['role'].startswith('post:')]==['osdi2023']
        tree = ast.parse((ROOT/'scripts/analyze.py').read_text())
        venues = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VENUES' for t in n.targets))
        with gzip.open(ROOT/'results/corpus_validation/features_numeric.jsonl.gz','rt') as handle: base=[json.loads(x) for x in handle]
        def metric(row,key): return row['op_additionally_1k']+row['op_notably_1k'] if key=='addnot_1k' else row[key]
        estimates = {'frozen':{},'with_recovered':{}}
        for mode, source in [('frozen',base),('with_recovered',base+additions)]:
            differences = collections.defaultdict(list)
            for venue, (_,pre_cells,post_cells) in venues.items():
                if '*' in venue: continue
                pre = [r for r in source if r['corpus'] in pre_cells]
                post = [r for r in source if r['corpus'] in post_cells]
                for key in metrics:
                    delta = statistics.fmean(metric(r,key) for r in post)-statistics.fmean(metric(r,key) for r in pre)
                    differences[key].append(delta)
                    if venue in recovered['pre_post_affected_venues']:
                        assert math.isclose(delta,recovered['pre_post_affected_venues'][venue][mode][key],abs_tol=1e-12)
            for key, values in differences.items():
                estimates[mode][key]=statistics.fmean(values)
                assert len(values)==10 and math.isclose(estimates[mode][key],recovered['ten_venue_equal_weight_mean_of_venue_contrasts'][mode][key],abs_tol=1e-12)
        changes={k:estimates['with_recovered'][k]-estimates['frozen'][k] for k in metrics}
        out['recovery_audit']={'admitted':7,'post':6,'series_only':1,'independent_equal_weight_simple_contrast_change':changes,'scope':'simple original-CS pre/post gaps only; not trend index or changepoint'}
        out['input_hashes'][str(recovery_path.relative_to(ROOT))]=digest(recovery_path)
    dest = ROOT/'results/review_round2/lead_verification.json'
    dest.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('Independent counts, estimates, source/code hashes and optional annotation contracts passed; no prose emitted.')
if __name__ == '__main__': main()
