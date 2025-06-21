from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

class LlamaSimplifier:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 cache_dir: str = "/scratch-shared/tpapandroeu/hf_cache"):
        """Initialize the Llama model for text simplification"""
        token = os.getenv("HUGGING_FACE_TOKEN")
        if not token:
            raise ValueError("HUGGING_FACE_TOKEN environment variable not set")

        # Clear CUDA cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load tokenizer and model directly
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=token)
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "token": token,
            "cache_dir": cache_dir,
            "trust_remote_code": True
        }
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.device = device
        self.model.eval()

        # Generation config
        self.generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3
        }
        self.batch_size = 8

    def _postprocess_output(self, text: str) -> str:
        # Remove lines starting with unwanted keywords
        unwanted_starts = ("Answer:", "Explanation:", "Corrected Answer:", "So my final choice", "Option", "Note:")
        lines = text.splitlines()
        filtered = [line for line in lines if not line.strip().startswith(unwanted_starts)]
        # Join back to a single string for further processing
        cleaned = " ".join(filtered)
        # Remove text after certain markers
        for marker in ["-no-", "-or-", "Corrected text", "was changed to", "was deleted", "is incorrect", "Explanation:"]:
            idx = cleaned.lower().find(marker)
            if idx != -1:
                cleaned = cleaned[:idx]
        # Remove bracketed or parenthetical comments
        cleaned = re.sub(r"\[.*?\]|\(.*?\)", "", cleaned)
        # Keep only the first sentence (ends with '.', '!' or '?')
        match = re.search(r"(.+?[.!?])", cleaned)
        if match:
            cleaned = match.group(1)
        # Strip whitespace
        return cleaned.strip()

    def simplify(self, prompt):
        # Single prompt inference
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_config
            )
        
        # Extract only the newly generated tokens
        generated_tokens = outputs[0][input_length:]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return self._postprocess_output(decoded)

    def simplify_batch(self, prompts: List[str]) -> List[str]:
        # Batch inference for a list of prompts
        results = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.model.device)
            input_lengths = [len(self.tokenizer.encode(p, add_special_tokens=True)) for p in batch]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config
                )
            
            # Extract only the newly generated tokens for each item
            batch_results = []
            for j, output in enumerate(outputs):
                generated_tokens = output[input_lengths[j]:]
                decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                batch_results.append(self._postprocess_output(decoded))
            
            results.extend(batch_results)
        return results

class MedicineLlamaSimplifier:
    def __init__(self, model_name: str = "instruction-pretrain/medicine-Llama3-8B", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 cache_dir: str = "/scratch-shared/tpapandroeu/hf_cache"):
        """Initialize the Medicine Llama model for text simplification"""
        token = os.getenv("HUGGING_FACE_TOKEN")
        if not token:
            raise ValueError("HUGGING_FACE_TOKEN environment variable not set")

        # Clear CUDA cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load tokenizer and model directly
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=token)
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "token": token,
            "cache_dir": cache_dir,
            "trust_remote_code": True
        }
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.device = device
        self.model.eval()

        # Generation config
        self.generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3
        }
        self.batch_size = 8

    def _postprocess_output(self, text: str) -> str:
        # Remove lines starting with unwanted keywords
        unwanted_starts = ("Answer:", "Explanation:", "Corrected Answer:", "So my final choice", "Option", "Note:")
        lines = text.splitlines()
        filtered = [line for line in lines if not line.strip().startswith(unwanted_starts)]
        # Join back to a single string for further processing
        cleaned = " ".join(filtered)
        # Remove text after certain markers
        for marker in ["-no-", "-or-", "Corrected text", "was changed to", "was deleted", "is incorrect", "Explanation:"]:
            idx = cleaned.lower().find(marker)
            if idx != -1:
                cleaned = cleaned[:idx]
        # Remove bracketed or parenthetical comments
        cleaned = re.sub(r"\[.*?\]|\(.*?\)", "", cleaned)
        # Keep only the first sentence (ends with '.', '!' or '?')
        match = re.search(r"(.+?[.!?])", cleaned)
        if match:
            cleaned = match.group(1)
        # Strip whitespace
        return cleaned.strip()

    def simplify(self, prompt):
        # Single prompt inference
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_config
            )
        
        # Extract only the newly generated tokens
        generated_tokens = outputs[0][input_length:]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return self._postprocess_output(decoded)

    def simplify_batch(self, prompts: List[str]) -> List[str]:
        # Batch inference for a list of prompts
        results = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.model.device)
            input_lengths = [len(self.tokenizer.encode(p, add_special_tokens=True)) for p in batch]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config
                )
            
            # Extract only the newly generated tokens for each item
            batch_results = []
            for j, output in enumerate(outputs):
                generated_tokens = output[input_lengths[j]:]
                decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                batch_results.append(self._postprocess_output(decoded))
            
            results.extend(batch_results)
        return results

def get_model(model_name: str, cache_dir: Optional[str] = None):
    """Factory function to get the appropriate model"""
    if model_name == "instruction-pretrain/medicine-Llama3-8B":
        return MedicineLlamaSimplifier(model_name=model_name, cache_dir=cache_dir)
    elif "llama" in model_name.lower():
        return LlamaSimplifier(
            model_name=model_name, 
            cache_dir=cache_dir
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")