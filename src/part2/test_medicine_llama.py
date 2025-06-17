import unittest
from models import get_model

class TestMedicineLlama(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the model once for all tests"""
        cls.model = get_model("instruction-pretrain/medicine-Llama3-8B")
    
    def test_simple_medical_text(self):
        """Test simplification of a simple medical text"""
        text = "The patient exhibited tachycardia with a heart rate of 150 beats per minute."
        simplified = self.model.simplify(text)
        print(f"\nOriginal: {text}")
        print(f"Simplified: {simplified}")
        self.assertIsNotNone(simplified)
        self.assertNotEqual(simplified, "")
    
    def test_complex_medical_text(self):
        """Test simplification of a more complex medical text"""
        text = ("The patient presented with acute myocardial infarction accompanied by " 
                "severe substernal chest pain radiating to the left arm and jaw, " 
                "diaphoresis, and dyspnea.")
        simplified = self.model.simplify(text)
        print(f"\nOriginal: {text}")
        print(f"Simplified: {simplified}")
        self.assertIsNotNone(simplified)
        self.assertNotEqual(simplified, "")
    
    def test_batch_simplification(self):
        """Test batch simplification"""
        texts = [
            "The patient exhibited tachycardia.",
            "The patient was diagnosed with pneumonia.",
            "The patient showed signs of hypertension."
        ]
        simplified = self.model.batch_simplify(texts, batch_size=2)
        print("\nBatch Simplification Results:")
        for orig, simp in zip(texts, simplified):
            print(f"\nOriginal: {orig}")
            print(f"Simplified: {simp}")
        self.assertEqual(len(simplified), len(texts))
        self.assertTrue(all(s != "" for s in simplified))

def main():
    # Example direct usage
    print("Direct Usage Example:")
    model = get_model("instruction-pretrain/medicine-Llama3-8B", cache_dir="/scratch-shared/tpapandroeu/hf_cache")
    
    text = "The patient exhibited tachycardia with concurrent dyspnea and diaphoresis."
    simplified = model.simplify(text)
    
    print(f"\nOriginal: {text}")
    print(f"Simplified: {simplified}")
    
    # Run tests
    print("\nRunning Tests:")
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == "__main__":
    main() 