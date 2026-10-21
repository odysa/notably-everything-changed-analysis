"""Validate private, self-reported human review inputs; never infer accuracy."""
import argparse
import hashlib
import json
import re
from pathlib import Path

LABELS = {'interpretive', 'functional', 'other', 'uncertain'}
ROW_KEYS = {'item_id', 'clauses', 'uncertain', 'note'}
CLAUSE_KEYS = {'start', 'end', 'verb_start', 'verb_end', 'label', 'mainclause_attached', 'uncertain'}


def load_packet(path, expected_count=120):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    packet = {}
    for row in rows:
        item_id = row.get('item_id')
        if not isinstance(item_id, str) or not item_id or item_id in packet or not isinstance(row.get('sentence'), str):
            raise ValueError('Invalid or duplicate packet item')
        packet[item_id] = row['sentence']
    if len(packet) != expected_count:
        raise ValueError(f'Packet must contain exactly {expected_count} items')
    return packet, hashlib.sha256(raw).hexdigest()


def validate_review(review, packet, packet_sha256):
    """Return item-ID keyed rows; reject incomplete checkpoints as final submissions."""
    if not isinstance(review, dict):
        raise ValueError('Review envelope must be an object')
    if type(review.get('schema_version')) is not int or review['schema_version'] != 1:
        raise ValueError('Unsupported schema_version')
    if review.get('packet_sha256') != packet_sha256:
        raise ValueError('Packet SHA-256 does not match')
    if not isinstance(review.get('reviewer_id'), str) or not review['reviewer_id'].strip():
        raise ValueError('reviewer_id must be nonempty')
    if review.get('reviewer_type') != 'human_self_reported':
        raise ValueError('reviewer_type must be human_self_reported')
    rows = review.get('annotations')
    if not isinstance(rows, list):
        raise ValueError('annotations must be an array')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise ValueError('Invalid annotation fields')
        item_id = row['item_id']
        if not isinstance(item_id, str) or item_id not in packet or item_id in result:
            raise ValueError('Unknown or duplicate item_id')
        if type(row['uncertain']) is not bool or not isinstance(row['note'], str) or not isinstance(row['clauses'], list):
            raise ValueError('Incomplete annotation: uncertainty must be boolean; note text and clauses required')
        if row['uncertain'] and not row['note'].strip():
            raise ValueError('Sentence uncertainty requires a note')
        text = packet[item_id]
        seen = set()
        for clause in row['clauses']:
            if not isinstance(clause, dict) or set(clause) != CLAUSE_KEYS:
                raise ValueError('Invalid clause fields')
            start, end, vs, ve = (clause[k] for k in ('start', 'end', 'verb_start', 'verb_end'))
            if any(type(n) is not int for n in (start, end, vs, ve)) or not (0 <= start < vs < ve <= end <= len(text)):
                raise ValueError('Invalid Unicode character spans')
            if text[start] != ',' or not re.fullmatch(r'[A-Za-z]+ing', text[vs:ve], re.IGNORECASE):
                raise ValueError('Clause must start with comma and head must be an -ing word')
            if not isinstance(clause['label'], str) or clause['label'] not in LABELS or type(clause['uncertain']) is not bool:
                raise ValueError('Invalid clause label or uncertainty')
            if clause['mainclause_attached'] is not None and type(clause['mainclause_attached']) is not bool:
                raise ValueError('Attachment must be boolean or null')
            if (clause['label'] == 'uncertain' or clause['mainclause_attached'] is None) and not clause['uncertain']:
                raise ValueError('Unresolved label/attachment requires clause uncertainty')
            signature = (start, end, vs, ve)
            if signature in seen:
                raise ValueError('Duplicate clause span')
            seen.add(signature)
        result[item_id] = row
    if set(result) != set(packet):
        raise ValueError('Incomplete final review: every packet item must occur exactly once')
    return result


def _certain(row):
    return not row['uncertain'] and all(not c['uncertain'] and c['label'] != 'uncertain' and c['mainclause_attached'] is not None for c in row['clauses'])


def _signature(row):
    return sorted((c['start'], c['end'], c['verb_start'], c['verb_end'], c['label'], c['mainclause_attached']) for c in row['clauses'])


def compare_reviews(a, b, packet, packet_sha256=None):
    """Agreement is exact span+label+attachment agreement, not precision or recall."""
    if packet_sha256 is None:
        raise ValueError('Independently loaded packet SHA-256 is required')
    digest = packet_sha256
    aa, bb = validate_review(a, packet, digest), validate_review(b, packet, digest)
    if a['reviewer_id'].strip() == b['reviewer_id'].strip():
        raise ValueError('Two different reviewer IDs are required')
    joint, equal, queue = 0, 0, []
    uncertainty_equal = 0
    for item_id in packet:
        x, y = aa[item_id], bb[item_id]
        uncertainty_equal += x['uncertain'] == y['uncertain']
        certain = _certain(x) and _certain(y)
        same = _signature(x) == _signature(y)
        joint += certain
        equal += certain and same
        if not certain or not same:
            queue.append({'item_id': item_id, 'reason': 'uncertainty' if not certain else 'disagreement', 'a': x, 'b': y})
    return {
        'schema_version': 1, 'packet_sha256': digest,
        'reviewer_ids': [a['reviewer_id'], b['reviewer_id']],
        'provenance': 'human_self_reported; identity, independence, and human authorship not independently verified',
        'interpretation': 'Validated label inputs and descriptive agreement only; not human validation, accuracy, precision, or recall. Joint certainty excludes any sentence/clause uncertainty or unresolved attachment. Empty clause lists count as certain negative judgments.',
        'items': len(packet), 'jointly_certain_items': joint, 'excluded_uncertain_items': len(packet) - joint,
        'exact_agreement_items': equal, 'exact_agreement_fraction': equal / joint if joint else None,
        'sentence_uncertainty_agreement_items': uncertainty_equal,
        'adjudication_queue': queue,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', required=True)
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--output', required=True, help='Private aggregate JSON destination, outside the repository')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    repository = Path(__file__).resolve().parents[3]
    if output == repository or repository in output.parents:
        parser.error('Output must be outside the repository; labels and adjudication remain private')
    packet, digest = load_packet(args.packet)
    try:
        result = compare_reviews(json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text()), packet, digest)
    except (ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    output.chmod(0o600)
    print('Both complete review inputs validated. Private agreement/adjudication output written.')


if __name__ == '__main__':
    main()
