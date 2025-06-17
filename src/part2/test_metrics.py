import unittest
from easse_metrics.sari import  compute_sari, extract_ngrams
from easse_metrics.bertscore import corpus_bertscore
from simplify import calculate_sentence_sari, calculate_sentence_bertscore, calculate_metrics

class TestMetrics(unittest.TestCase):
    def debug_sari_calculation(self, orig: str, sys: str, ref: str):
        """Helper function to debug SARI calculation"""
        print("\nDEBUG SARI calculation:")
        print(f"Original: {orig}")
        print(f"System: {sys}")
        print(f"Reference: {ref}")
        
        # Debug n-grams
        print("\nN-grams:")
        orig_ngrams = extract_ngrams(orig)
        sys_ngrams = extract_ngrams(sys)
        ref_ngrams = extract_ngrams(ref)
        
        print("Original n-grams:", dict(orig_ngrams))
        print("System n-grams:", dict(sys_ngrams))
        print("Reference n-grams:", dict(ref_ngrams))
    
    def test_sari_perfect_match(self):
        """Test SARI score for perfect matches"""
        orig = "The patient exhibited tachycardia."
        sys = "The patient had a fast heartbeat."
        ref = "The patient had a fast heartbeat."
        score = compute_sari(orig, sys, ref)
        self.assertGreaterEqual(score, 90.0)  # Should be very high
    
    def test_sari_no_match(self):
        """Test SARI score for completely different sentences"""
        orig = "The patient exhibited tachycardia."
        sys = "The patient exhibited tachycardia."  # No simplification
        ref = "The patient had a fast heartbeat."
        score = compute_sari(orig, sys, ref)
        self.assertLess(score, 50.0)  # Should be low since no simplification was done
    
    def test_sari_partial_match(self):
        """Test SARI score for partial matches"""
        orig = "The patient exhibited tachycardia and dyspnea."
        sys = "The patient had a fast heartbeat and breathing problems."
        ref = "The patient had a rapid heart rate and trouble breathing."
        score = compute_sari(orig, sys, ref)
        self.assertGreater(score, 50.0)  # Should be relatively high
        self.assertLess(score, 100.0)    # But not perfect

class TestSentenceMetrics(unittest.TestCase):
    def setUp(self):
        """Set up some example sentences for testing"""
        self.examples = {
            'perfect_match': {
                'original': "The patient exhibited tachycardia.",
                'simplified': "The patient had a fast heartbeat.",
                'reference': "The patient had a fast heartbeat.",
                'expected': {
                    'sari': {'min': 90.0, 'max': 100.0},      # High SARI for good simplification
                    'bertscore': {'min': 90.0, 'max': 100.0}  # High BERTScore for matching reference
                }
            },
            'no_match': {
                'original': "The patient exhibited tachycardia.",
                'simplified': "The patient exhibited tachycardia.",
                'reference': "The patient had a fast heartbeat.",
                'expected': {
                    'sari': {'min': 0.0, 'max': 50.0},       # Low SARI for no simplification
                    'bertscore': {'min': 50.0, 'max': 80.0}   # Moderate BERTScore due to semantic similarity
                }
            },
            'partial_match': {
                'original': "The patient exhibited tachycardia and dyspnea.",
                'simplified': "The patient had a fast heartbeat and breathing problems.",
                'reference': "The patient had a rapid heart rate and trouble breathing.",
                'expected': {
                    'sari': {'min': 50.0, 'max': 90.0},      # Moderate SARI for partial match
                    'bertscore': {'min': 80.0, 'max': 95.0}   # High BERTScore due to semantic similarity
                }
            }
        }

    def test_sentence_sari(self):
        """Test SARI calculation at sentence level"""
        print("\nTesting SARI scores:")
        for case_name, case in self.examples.items():
            score = calculate_sentence_sari(
                case['original'],
                case['simplified'],
                case['reference']
            )
            print(f"\n{case_name}:")
            print(f"Original: {case['original']}")
            print(f"Simplified: {case['simplified']}")
            print(f"Reference: {case['reference']}")
            print(f"SARI score: {score:.2f}")
            
            self.assertGreaterEqual(score, case['expected']['sari']['min'])
            self.assertLessEqual(score, case['expected']['sari']['max'])

    def test_sentence_bertscore(self):
        """Test BERTScore calculation at sentence level"""
        print("\nTesting BERTScore scores:")
        for case_name, case in self.examples.items():
            score = calculate_sentence_bertscore(
                case['simplified'],
                case['reference']
            )
            print(f"\n{case_name}:")
            print(f"Simplified: {case['simplified']}")
            print(f"Reference: {case['reference']}")
            print(f"BERTScore: {score:.2f}")
            
            self.assertGreaterEqual(score, case['expected']['bertscore']['min'])
            self.assertLessEqual(score, case['expected']['bertscore']['max'])

    def test_combined_metrics(self):
        """Test all metrics together using calculate_metrics"""
        print("\nTesting combined metrics calculation:")
        for case_name, case in self.examples.items():
            metrics = calculate_metrics(
                case['original'],
                case['simplified'],
                case['reference'],
                debug=True
            )
            print(f"\n{case_name}:")
            print(f"Original: {case['original']}")
            print(f"Simplified: {case['simplified']}")
            print(f"Reference: {case['reference']}")
            print(f"Metrics: {metrics}")
            
            self.assertGreaterEqual(metrics['sari'], case['expected']['sari']['min'])
            self.assertLessEqual(metrics['sari'], case['expected']['sari']['max'])
            self.assertGreaterEqual(metrics['bertscore'], case['expected']['bertscore']['min'])
            self.assertLessEqual(metrics['bertscore'], case['expected']['bertscore']['max'])

    def test_edge_cases(self):
        """Test edge cases for sentence-level metrics"""
        edge_cases = {
            'empty': {
                'original': "",
                'simplified': "",
                'reference': ""
            },
            'single_word': {
                'original': "tachycardia",
                'simplified': "heartbeat",
                'reference': "heartbeat"
            },
            'very_long': {
                'original': "The patient " * 20 + "exhibited tachycardia.",
                'simplified': "The patient " * 20 + "had a fast heartbeat.",
                'reference': "The patient " * 20 + "had a fast heartbeat."
            }
        }
        
        print("\nTesting edge cases:")
        for case_name, case in edge_cases.items():
            print(f"\n{case_name}:")
            metrics = calculate_metrics(
                case['original'],
                case['simplified'],
                case['reference'],
                debug=True
            )
            print(f"Metrics: {metrics}")
            
            # Basic sanity checks for edge cases
            self.assertGreaterEqual(metrics['sari'], 0.0)
            self.assertLessEqual(metrics['sari'], 100.0)
            self.assertGreaterEqual(metrics['bertscore'], 0.0)
            self.assertLessEqual(metrics['bertscore'], 100.0)

def main():
    unittest.main(verbosity=2)

if __name__ == "__main__":
    main() 