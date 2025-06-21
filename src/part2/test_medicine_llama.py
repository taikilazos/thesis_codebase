import unittest
import transformers
import torch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TestMedicineLlama(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the model once for all tests"""
        model_id = "instruction-pretrain/medicine-Llama3-8B"
        token = os.getenv("HUGGING_FACE_TOKEN")
        if not token:
            raise ValueError("HUGGING_FACE_TOKEN environment variable not set")
        
        print("Initializing model and tokenizer...")
        cls.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=token)
        cls.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token
        )
        cls.model.eval()
        
        # Generation config
        cls.generation_config = {
            "max_new_tokens": 256,
            "temperature": 0.2,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3
        }
    
    def simplify_text(self, text):
        """Helper method to simplify text using the model"""
        prompt = f"Simplify this medical text: {text}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_config
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the response part (after the prompt)
        response = decoded[len(prompt):].strip()
        return response
    
    def test_simple_medical_text(self):
        """Test simplification of a simple medical text"""
        text = "The patient exhibited tachycardia with a heart rate of 150 beats per minute."
        simplified = self.simplify_text(text)
        print(f"\nOriginal: {text}")
        print(f"Simplified: {simplified}")
        self.assertIsNotNone(simplified)
        self.assertNotEqual(simplified, "")
    
    def test_complex_medical_text(self):
        """Test simplification of a more complex medical text"""
        text = ("The patient presented with acute myocardial infarction accompanied by " 
                "severe substernal chest pain radiating to the left arm and jaw, " 
                "diaphoresis, and dyspnea.")
        simplified = self.simplify_text(text)
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
        simplified = []
        for text in texts:
            result = self.simplify_text(text)
            simplified.append(result)
        
        print("\nBatch Simplification Results:")
        for orig, simp in zip(texts, simplified):
            print(f"\nOriginal: {orig}")
            print(f"Simplified: {simp}")
        self.assertEqual(len(simplified), len(texts))
        self.assertTrue(all(s != "" for s in simplified))

def test_medicine_llama_direct():
    """Direct usage example following test_llama.py pattern"""
    model_id = "instruction-pretrain/medicine-Llama3-8B"
    token = os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        raise ValueError("HUGGING_FACE_TOKEN environment variable not set")
    
    print("Initializing model and tokenizer (direct)...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=token)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token
    )
    
    text = "The patient exhibited tachycardia with concurrent dyspnea and diaphoresis."
    prompt = f"Simplify this medical text: {text}"
    
    print("Prompt to model (direct):")
    print(prompt)
    print("Generating response (direct)...")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded[len(prompt):].strip()
    
    print("\nModel output (direct):")
    print(f"Original: {text}")
    print(f"Simplified: {response}")

def main():
    # Example direct usage
    print("Direct Usage Example:")
    test_medicine_llama_direct()
    
    # Run tests
    print("\nRunning Tests:")
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == "__main__":
    main() 