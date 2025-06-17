from typing import List, Optional
import torch
import re
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch.cuda.amp as amp
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class LlamaSimplifier:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 cache_dir: str = "/scratch-shared/tpapandroeu/hf_cache",
                 use_quantization: bool = True,
                 use_flash_attention: bool = True):
        """Initialize the Llama model for text simplification"""
        # Get token from environment variable
        token = os.getenv("HUGGING_FACE_TOKEN")
        if not token:
            raise ValueError("HUGGING_FACE_TOKEN environment variable not set")
        
        # Clear CUDA cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Configure mixed precision
        self.scaler = amp.GradScaler('cuda' if device == 'cuda' else 'cpu')
        self.autocast_context = lambda: torch.amp.autocast(device_type=self.device_name, dtype=torch.float16)
        
        # Configure model loading
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "token": token,
            "trust_remote_code": True
        }
        
        # Try to add Flash Attention if requested
        if use_flash_attention:
            try:
                # Use the recommended parameter name
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("Flash Attention enabled")
            except Exception as e:
                print(f"Could not enable Flash Attention: {str(e)}")
        
        # Try to add Quantization if requested
        if use_quantization:
            try:
                # Only import if quantization is requested
                from transformers import BitsAndBytesConfig
                
                print("BitsAndBytes available, enabling 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                model_kwargs["quantization_config"] = quantization_config
            except ImportError:
                print("BitsAndBytes not available, falling back to full precision")
            except Exception as e:
                print(f"Error setting up quantization: {str(e)}")
        
        # Initialize the pipeline with proper configuration
        self.pipeline = pipeline(
            task="text-generation",
            model=model_name,
            cache_dir=cache_dir,
            **model_kwargs
        )
        
        # Ensure tokenizer is properly configured
        if not hasattr(self.pipeline.tokenizer, 'pad_token') or self.pipeline.tokenizer.pad_token is None:
            self.pipeline.tokenizer.pad_token = self.pipeline.tokenizer.eos_token
        
        self.device = device
        
        # Generation config
        self.generation_config = {
            "max_new_tokens": 512,  # Reduced from 1024
            "temperature": 0.2,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3
        }
        
        # Set batch size
        self.batch_size = 8

    def _clean_response(self, response: str) -> str:
        """Extract simplified text from model output"""
        if not response:
            return ""
        
        # First priority: Look for Final Answer pattern
        if "**Final Answer:**" in response:
            parts = response.split("**Final Answer:**")
            if len(parts) > 1:
                return parts[1].strip().split("\n")[0].strip()
        
        # Second priority: Look for other simplification patterns
        patterns = [
            "Option A -", 
            "Here's your rewritten text:",
            "Rewritten as",
            "Simplified version:",
            "Write one simplified sentence:"
        ]
        
        for pattern in patterns:
            if pattern in response:
                parts = response.split(pattern)
                if len(parts) > 1:
                    # Take the first non-empty line after the pattern
                    for line in parts[1].split("\n"):
                        line = line.strip()
                        if line and len(line) > 10 and not line.startswith("Option"):
                            return line
        
        # If we can't find any pattern matches, don't return the original!
        # Return an empty string or error message instead
        return "EXTRACTION_FAILED"

    # Add a new method to access the model directly
    def generate_direct(self, prompts, **kwargs):
        """Generate text directly using the model, bypassing pipeline parameter issues"""
        results = []
        
        # Access the tokenizer and model directly
        tokenizer = self.pipeline.tokenizer
        model = self.pipeline.model
        
        # Define allowed generation parameters
        generation_params = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 0.3),
            "top_p": kwargs.get("top_p", 0.95),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.3),
        }
        
        print(f"Starting generation with params: {generation_params}")
        
        # Process in batches
        batch_size = 4  # Process 4 prompts at a time
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            # Tokenize all prompts in the batch
            batch_inputs = tokenizer(batch, padding=True, return_tensors="pt").to("cuda")
            
            # Generate responses
            with torch.no_grad():
                output_ids = model.generate(
                    **batch_inputs,
                    **generation_params
                )
                
                # Decode each output in the batch
                for j in range(len(batch)):
                    # Extract the individual sequence (handle both formats)
                    if output_ids.ndim == 3:  # [batch, beam, seq_len]
                        sequence = output_ids[j, 0, :]  # First beam for each batch item
                    else:  # [batch, seq_len]
                        sequence = output_ids[j, :]
                    
                    # Decode the full generated text WITHOUT removing the prompt
                    full_decoded = tokenizer.decode(sequence, skip_special_tokens=True)
                    
                    # Get just the response part (for debugging, keep the prompt part too)
                    results.append(full_decoded)
                    
    
        return results

def get_model(model_name: str, cache_dir: Optional[str] = None, use_flash_attention: bool = True, use_quantization: bool = True):
    """Factory function to get the appropriate model"""
    if model_name == "instruction-pretrain/medicine-Llama3-8B":
        # MedicineLlamaSimplifier implementation would go here
        return LlamaSimplifier(model_name=model_name, cache_dir=cache_dir)
    elif "llama" in model_name.lower():
        return LlamaSimplifier(
            model_name=model_name, 
            cache_dir=cache_dir, 
            use_flash_attention=use_flash_attention,
            use_quantization=use_quantization
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")