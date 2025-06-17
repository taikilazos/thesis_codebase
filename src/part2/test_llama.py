import transformers
import torch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_llama():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Get token from environment variable
    token = os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        raise ValueError("HUGGING_FACE_TOKEN environment variable not set")
    
    print("Initializing pipeline...")
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        },
        token=token  # Use token from environment variable
    )
    
    print("Creating test message...")
    messages = [
        {"role": "system", "content": "You are a helpful medical text simplification assistant."},
        {"role": "user", "content": "Please simplify this medical text: 'Patients with chronic obstructive pulmonary disease may experience dyspnea.'"},
    ]
    
    print("Generating response...")
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    print("\nModel output:")
    print(outputs[0]["generated_text"][2]['content'])

if __name__ == "__main__":
    test_llama() 