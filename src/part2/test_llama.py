import transformers
import torch
import os
from dotenv import load_dotenv
from prompts import get_prompt_function

# Load environment variables
load_dotenv()

def test_llama_direct():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
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
    prompt_func = get_prompt_function("simple")
    sentence = "Patients with chronic obstructive pulmonary disease may experience dyspnea."
    prompt = prompt_func(sentence)
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
    print("\nModel output (direct):")
    print(decoded)

if __name__ == "__main__":
    test_llama_direct() 