import unittest
from easse_metrics.sari import compute_sari
from easse_metrics.bleu import corpus_bleu
from easse_metrics.fkgl import corpus_fkgl
from easse_metrics.bertscore import corpus_bertscore

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.examples = {
            'perfect_match': {
                'original': "The patient exhibited tachycardia.",
                'simplified': "The patient had a fast heartbeat.",
                'reference': "The patient had a fast heartbeat.",
            },
            'no_match': {
                'original': "The patient exhibited tachycardia.",
                'simplified': "The patient exhibited tachycardia.",
                'reference': "The patient had a fast heartbeat.",
            },
            'partial_match': {
                'original': "The patient exhibited tachycardia and dyspnea.",
                'simplified': "The patient had a fast heartbeat and breathing problems.",
                'reference': "The patient had a rapid heart rate and trouble breathing.",
            },
            'empty': {
                'original': "",
                'simplified': "",
                'reference': "",
            },
            'single_word': {
                'original': "tachycardia",
                'simplified': "heartbeat",
                'reference': "heartbeat",
            },
            'very_long': {
                'original': "The patient " * 20 + "exhibited tachycardia.",
                'simplified': "The patient " * 20 + "had a fast heartbeat.",
                'reference': "The patient " * 20 + "had a fast heartbeat.",
            }
        }

    def test_sari(self):
        print("\nTesting SARI:")
        for name, case in self.examples.items():
            sari = compute_sari(case['original'], case['simplified'], case['reference'])
            print(f"{name}: SARI = {sari:.2f}")
            # Assert reasonable ranges
            self.assertGreaterEqual(sari, 0.0)
            self.assertLessEqual(sari, 100.0)
            if name == 'perfect_match':
                self.assertGreaterEqual(sari, 80.0)
            if name == 'no_match':
                self.assertLess(sari, 60.0)

    def test_bleu(self):
        print("\nTesting BLEU:")
        for name, case in self.examples.items():
            bleu = corpus_bleu([case['simplified']], [[case['reference']]])
            print(f"{name}: BLEU = {bleu:.2f}")
            self.assertGreaterEqual(bleu, 0.0)
            self.assertLessEqual(bleu, 100.0001)
            if name == 'perfect_match':
                self.assertGreaterEqual(bleu, 80.0)
            if name == 'no_match':
                self.assertLess(bleu, 60.0)

    def test_fkgl(self):
        print("\nTesting FKGL:")
        for name, case in self.examples.items():
            fkgl = corpus_fkgl([case['simplified']])
            print(f"{name}: FKGL = {fkgl:.2f}")
            self.assertGreaterEqual(fkgl, 0.0)
            # No strict upper bound, but should be reasonable
            if name == 'empty':
                self.assertEqual(fkgl, 0.0)

    def test_bertscore(self):
        print("\nTesting BERTScore:")
        for name, case in self.examples.items():
            f1 = corpus_bertscore([case['simplified']], [[case['reference']]])
            print(f"{name}: BERTScore F1 = {f1:.2f}")
            self.assertGreaterEqual(f1, 0.0)
            self.assertLessEqual(f1, 100.0)
            if name == 'perfect_match':
                self.assertGreaterEqual(f1, 80.0)
            if name == 'no_match':
                self.assertLess(f1, 80.0)

if __name__ == "__main__":
    unittest.main(verbosity=2) 