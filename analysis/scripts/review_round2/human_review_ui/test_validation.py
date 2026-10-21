"""Invented fixtures only; no private packet prose or reviewer labels."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('validation', Path(__file__).with_name('validation.py'))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.text = 'We test λ, showing a result.'
        self.packet = {'fictional': self.text}
        self.review = {'schema_version': 1, 'packet_sha256': 'abc', 'reviewer_id': 'A', 'reviewer_type': 'human_self_reported', 'annotations': [{'item_id': 'fictional', 'clauses': [{'start': 9, 'end': len(self.text), 'verb_start': 11, 'verb_end': 18, 'label': 'functional', 'mainclause_attached': True, 'uncertain': False}], 'uncertain': False, 'note': ''}]}

    def validate(self):
        return v.validate_review(self.review, self.packet, 'abc')

    def test_valid_unicode_and_metadata(self):
        self.review['progress'] = {'completed': 1}
        self.assertEqual(len(self.validate()), 1)

    def test_bad_hash(self):
        self.review['packet_sha256'] = 'different'
        with self.assertRaises(ValueError): self.validate()

    def test_duplicate(self):
        self.review['annotations'] *= 2
        with self.assertRaises(ValueError): self.validate()

    def test_missing(self):
        self.review['annotations'] = []
        with self.assertRaisesRegex(ValueError, 'Incomplete'): self.validate()

    def test_checkpoint(self):
        self.review['annotations'][0]['uncertain'] = None
        with self.assertRaises(ValueError): self.validate()

    def test_bad_spans_and_boolean_integer(self):
        for key, value in [('start', True), ('end', 999), ('verb_start', 8), ('verb_end', 999)]:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.review)
                bad['annotations'][0]['clauses'][0][key] = value
                with self.assertRaises(ValueError): v.validate_review(bad, self.packet, 'abc')

    def test_non_ing_and_noncomma(self):
        for text in [self.text.replace('showing', 'followe'), self.text.replace(',', ';')]:
            with self.assertRaises(ValueError): v.validate_review(self.review, {'fictional': text}, 'abc')

    def test_uncertainty_consistency(self):
        clause = self.review['annotations'][0]['clauses'][0]
        clause['mainclause_attached'] = None
        with self.assertRaises(ValueError): self.validate()
        clause['uncertain'] = True
        self.validate()
        other = copy.deepcopy(self.review)
        other['reviewer_id'] = 'B'
        aggregate = v.compare_reviews(self.review, other, self.packet, 'abc')
        self.assertEqual(aggregate['excluded_uncertain_items'], 1)
        self.assertIsNone(aggregate['exact_agreement_fraction'])

    def test_same_reviewer_rejected(self):
        with self.assertRaises(ValueError): v.compare_reviews(self.review, copy.deepcopy(self.review), self.packet, 'abc')

    def test_agreement_and_disagreement(self):
        other = copy.deepcopy(self.review)
        other['reviewer_id'] = 'B'
        self.assertEqual(v.compare_reviews(self.review, other, self.packet, 'abc')['exact_agreement_fraction'], 1)
        other['annotations'][0]['clauses'][0]['label'] = 'interpretive'
        result = v.compare_reviews(self.review, other, self.packet, 'abc')
        self.assertEqual(result['exact_agreement_fraction'], 0)
        self.assertEqual(len(result['adjudication_queue']), 1)

    def test_packet_size_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'packet.jsonl'
            path.write_text('{"item_id":"fictional","sentence":"Invented sentence."}\n')
            with self.assertRaises(ValueError): v.load_packet(path)
            packet, digest = v.load_packet(path, expected_count=1)
            self.assertEqual(len(digest), 64)
            self.assertEqual(len(packet), 1)


if __name__ == '__main__':
    unittest.main()
