# Comment out these lines at the top:
# import multiprocessing
# Set the start method to spawn for CUDA compatibility
# multiprocessing.set_start_method('spawn', force=True)

# Rest of your imports...
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import nltk
from typing import Dict, List, Tuple
from tqdm import tqdm
import sys
import json
import os
from dotenv import load_dotenv
import torch.cuda.amp as amp
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from functools import partial
import time
import re

# Import our model and prompt functions
from models import get_model
from prompts import create_jargon_prompt

# Load environment variables
load_dotenv()

# Define a dataset for batch processing
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.device_name = 'cuda' if device == 'cuda' else 'cpu'
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

class CLEF2025Pipeline:
    def __init__(self, 
                 jargon_model_path: str = "/home/tpapandroeu/reproducibility/output/medreadme/roberta_large_binary.pt",  # Changed to binary model
                 simplifier_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 hf_token: str = None,
                 batch_size: int = 32,
                 cache_dir: str = "/scratch-shared/tpapandroeu/hf_cache",
                 debug: bool = True,
                 use_flash_attention: bool = True,
                 use_quantization: bool = True,
                 num_workers: int = 4):
        """
        Initialize the CLEF 2025 pipeline with both jargon detection and simplification models
        """
        self.debug = debug
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug_print("Initializing CLEF2025Pipeline...")
        
        try:
            # Set up device and mixed precision
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.debug_print(f"Using device: {self.device}")
            self.scaler = amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
            self.device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.debug_print(f"Available CUDA memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
            
            # Initialize binary jargon detection model
            self.debug_print(f"Loading binary jargon detection model from {jargon_model_path}...")
            self.jargon_tokenizer = AutoTokenizer.from_pretrained("roberta-large", add_prefix_space=True)
            self.debug_print("Loaded jargon tokenizer")
            
            self.jargon_model = AutoModelForTokenClassification.from_pretrained(
                "roberta-large", 
                num_labels=2,  # Changed to 2 for binary classification
                torch_dtype=torch.bfloat16
            )
            self.debug_print("Created jargon model architecture")
            
            state_dict = torch.load(jargon_model_path)
            self.jargon_model.load_state_dict(state_dict)
            self.debug_print("Loaded jargon model weights successfully")
            
            self.jargon_model.to(self.device)
            self.jargon_model.eval()
            
            # Initialize simplifier model
            self.debug_print(f"Loading simplifier model {simplifier_model_name}...")
            self.simplifier = get_model(
                simplifier_model_name, 
                cache_dir=cache_dir,
                use_flash_attention=use_flash_attention,
                use_quantization=use_quantization
            )
            self.debug_print("Simplifier model loaded successfully")
            
            # Updated for binary classification
            self.id2label = {0: 'O', 1: 'JARGON'}
            
            nltk.download('punkt', quiet=True)
            self.debug_print("Pipeline initialization completed successfully")
            
        except Exception as e:
            self.debug_print(f"Error during pipeline initialization: {str(e)}", error=True)
            raise
    
    def get_autocast_context(self):
        """Return the autocast context for mixed precision"""
        # Fixed version that passes device_type properly
        return torch.amp.autocast(device_type=self.device_name, dtype=torch.float16)

    def debug_print(self, message: str, error: bool = False):
        if self.debug:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output = f"[{timestamp}] {'ERROR: ' if error else ''}{message}"
            print(output, file=sys.stderr if error else sys.stdout, flush=True)

    def _process_single_text_jargons(self, text: str) -> Dict[str, List[str]]:
        """Process a single text for binary jargon detection"""
        # Initialize jargons dict with a single category for jargons
        text_jargons = {'JARGON': set()}
        
        WINDOW_SIZE = 200
        OVERLAP = 50
        
        # Split text into sentences
        sentences = nltk.sent_tokenize(text)
        
        # Process each sentence
        for sentence in sentences:
            # Tokenize sentence
            tokens = nltk.word_tokenize(sentence)
            
            # Skip empty sentences
            if not tokens:
                continue
            
            # Get total tokens for the sentence
            all_subtokens = self.jargon_tokenizer.tokenize(" ".join(tokens))
            
            # If sentence exceeds token limit, process it in windows
            if len(all_subtokens) > 250:
                # Process sentence in overlapping windows
                for start_idx in range(0, len(tokens), WINDOW_SIZE - OVERLAP):
                    window_tokens = tokens[start_idx:start_idx + WINDOW_SIZE]
                    
                    # Process window
                    encoding = self.jargon_tokenizer(
                        window_tokens,
                        is_split_into_words=True,
                        padding=True,
                        truncation=True,
                        max_length=250,
                        return_tensors='pt'
                    )
                    
                    self._process_window(encoding, window_tokens, text_jargons)
            else:
                # Process entire sentence if it's within limits
                encoding = self.jargon_tokenizer(
                    tokens,
                    is_split_into_words=True,
                    padding=True,
                    truncation=True,
                    max_length=250,
                    return_tensors='pt'
                )
                
                self._process_window(encoding, tokens, text_jargons)
        
        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in text_jargons.items()}

    def detect_jargons_batch(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        """
        Detect jargons in a batch of texts (sequential version)
        """
        self.debug_print(f"\n=== Starting Batch Jargon Detection for {len(texts)} texts ===")
        
        try:
            start_time = time.time()
            
            # Process texts sequentially, no multiprocessing
            all_jargons = []
            for text in tqdm(texts, desc="Detecting jargons"):
                jargons = self._process_single_text_jargons(text)
                all_jargons.append(jargons)
            
            end_time = time.time()
            self.debug_print(f"Jargon detection completed in {end_time - start_time:.2f} seconds")
            
            return all_jargons
            
        except Exception as e:
            self.debug_print(f"Error during jargon detection: {str(e)}", error=True)
            raise

    def _process_window(self, encoding, tokens, text_jargons):
        """Helper method to process a window of tokens for binary jargon detection"""
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get predictions with mixed precision for faster inference
        with torch.no_grad(), self.get_autocast_context():
            outputs = self.jargon_model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Process predictions
        current_term = []
        word_ids = encoding.word_ids(batch_index=0)
        
        for idx, (pred, word_id) in enumerate(zip(predictions[0], word_ids)):
            if word_id is not None:
                label = self.id2label[pred.item()]
                if label == 'JARGON':
                    token = tokens[word_id]
                    if not current_term:
                        current_term = [token]
                    else:
                        current_term.append(token)
                else:  # Not a jargon (label == 'O')
                    if current_term:
                        text_jargons['JARGON'].add(' '.join(current_term))
                        current_term = []
        
        # Don't forget the last term
        if current_term:
            text_jargons['JARGON'].add(' '.join(current_term))

    def create_simplification_prompt(self, text: str, jargons: Dict[str, List[str]]) -> str:
        """Create a prompt for the simplification model using binary detected jargons"""
        # Process jargon terms - with binary model we only have one category
        all_terms = set()
        if 'JARGON' in jargons and jargons['JARGON']:
            for term in jargons['JARGON']:
                all_terms.add(term)
        
        # Sort terms by length (longer terms first) and then alphabetically
        sorted_terms = sorted(all_terms, key=lambda x: (-len(x), x.lower()))
        
        # Create prompt using the jargon prompt function
        prompt = create_jargon_prompt(text, sorted_terms)
        
        # Debug print the first prompt only
        if self.debug and not hasattr(self, '_printed_first_prompt'):
            self.debug_print("\nExample prompt format:")
            self.debug_print(prompt + "\n")
            self._printed_first_prompt = True
            
        return prompt

    def generate_simplified_text_batch(self, prompts: List[str], max_new_tokens: int = 512) -> List[str]:
        """Generate simplified text using direct model access instead of pipeline"""
        self.debug_print(f"\n=== Generating Simplified Text for {len(prompts)} prompts ===")
        
        try:
            start_time = time.time()
            
            
            # Define generation parameters
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.3,  # Try increasing to 0.5 for more creative simplifications
                "top_p": 0.95,
                "do_sample": True,
                "repetition_penalty": 1.3
            }
            
            
            # Use direct model access instead of pipeline
            raw_results = self.simplifier.generate_direct(
                prompts,
                **generation_config
            )
            

            # Clean responses if needed
            cleaned_results = []
            for i, r in enumerate(raw_results):
                cleaned = self._clean_response(r) if r else ""
                if i == 0:  # Debug the first response cleaning
                    self.debug_print(f"\n=== CLEANED OUTPUT ===")
                    self.debug_print(f"BEFORE: {r[:100]}...")
                    self.debug_print(f"AFTER: {cleaned}")
                    self.debug_print("=== END CLEANED OUTPUT ===\n")
                cleaned_results.append(cleaned)
            
            end_time = time.time()
            self.debug_print(f"Text generation completed in {end_time - start_time:.2f} seconds")
            
            return cleaned_results
            
        except Exception as e:
            self.debug_print(f"Error during text generation: {str(e)}", error=True)
            import traceback
            self.debug_print(f"Full traceback: {traceback.format_exc()}", error=True)
            return [""] * len(prompts)

    def _clean_response(self, response: str) -> str:
        """Enhanced response cleaner with better artifact removal"""
        if not response:
            return ""
        
        # First look for Final Answer pattern
        if "**Final Answer:**" in response:
            parts = response.split("**Final Answer:**")
            if len(parts) > 1:
                simplified = parts[1].strip().split("\n")[0].strip()
                return self._post_process_cleaned_text(simplified)
        
        # Clean common artifacts
        simplified = ""
        patterns = [
            "Option A -", 
            "Here's your rewritten text:",
            "Rewritten as",
            "Simplified version:",
            "Write one simplified sentence:",
            "Simple:"
        ]
        
        for pattern in patterns:
            if pattern in response:
                parts = response.split(pattern)
                if len(parts) > 1:
                    for line in parts[1].split("\n"):
                        line = line.strip()
                        if line and len(line) > 10 and not line.startswith("Option"):
                            simplified = line
                            break
                
                if simplified:
                    return self._post_process_cleaned_text(simplified)
    
        # Fallback to finding first substantial line after the original text
        if "Text to simplify:" in response:
            parts = response.split("Text to simplify:")
            if len(parts) > 1 and "\n" in parts[1]:
                lines = parts[1].split("\n")
                for i in range(1, len(lines)):  # Skip the first line which is the original text
                    line = lines[i].strip()
                    if line and len(line) > 15:
                        return self._post_process_cleaned_text(line)
    
        return "EXTRACTION_FAILED"

    def _post_process_cleaned_text(self, text: str) -> str:
        """Clean up artifacts in extracted text"""
        # Remove instruction headers
        if text.startswith("IMPORTANT:"):
            parts = text.split("IMPORTANT:")
            text = parts[-1].strip()
        
        # Remove category labels
        text = re.sub(r'\s+Category:\s+[\w/]+\s*$', '', text)
        
        # Remove any term mappings
        text = re.sub(r'\s+-\s+[\w\s]+->', '', text)
        
        # Ensure text starts with uppercase letter
        if text and not text[0].isupper() and len(text) > 1:
            text = text[0].upper() + text[1:]
        
        return text

    def _preprocess_jargon_terms(self, jargons: Dict[str, List[str]]) -> List[str]:
        """Preprocess detected jargons to remove duplications and clean up terms"""
        all_terms = set()
        
        if 'JARGON' in jargons and jargons['JARGON']:
            for term in jargons['JARGON']:
                # Remove duplicated words (like "brachytherapy brachytherapy")
                words = term.split()
                cleaned_words = []
                prev_word = None
                for word in words:
                    if word != prev_word:
                        cleaned_words.append(word)
                        prev_word = word
                
                cleaned_term = " ".join(cleaned_words)
                all_terms.add(cleaned_term)
        
        # Sort terms by length (longer terms first) to prioritize multi-word jargons
        return sorted(all_terms, key=lambda x: (-len(x), x.lower()))

    def simplify_batch(self, texts: List[str]) -> List[str]:
        """
        Main pipeline function: detect jargons and simplify multiple texts at once
        Args:
            texts: List of input texts to simplify
        Returns:
            List of simplified texts
        """
        self.debug_print(f"\n=== Starting Batch Text Simplification Pipeline for {len(texts)} texts ===")
        
        try:
            start_time = time.time()
            
            # Step 1: Detect jargons in batch
            jargons_list = self.detect_jargons_batch(texts)

            print(f"Detected jargons: {jargons_list}")
            
            # Step 2: Create prompts (can be parallelized)
            prompts = [
                self.create_simplification_prompt(text, jargons)
                for text, jargons in zip(texts, jargons_list)
            ]
            print("PROMPTS!!!!!")
            print(prompts)  # Debug: Print the first prompt
            
            # Step 3: Generate simplified texts with optimized batch processing (cleaned)
            simplified_texts = self.generate_simplified_text_batch(prompts)

            print(simplified_texts)
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / len(texts)
            
            self.debug_print(f"Completed simplification pipeline in {total_time:.2f} seconds")
            self.debug_print(f"Average time per text: {avg_time:.2f} seconds")
            
            return simplified_texts
            
        except Exception as e:
            self.debug_print(f"Error in simplification pipeline: {str(e)}", error=True)
            raise

def main():
    try:
        # Get part number and task from command line
        if len(sys.argv) < 3:
            print("Usage: python clef2025_batch.py <task_num> <part_num>")
            print("Example: python clef2025_batch.py 11 1")
            sys.exit(1)
            
        task_num = sys.argv[1]
        part_num = int(sys.argv[2])
        
        if task_num not in ['11', '12']:
            print("Error: task_num must be either 11 or 12")
            sys.exit(1)
        
        # Create cache directory in scratch space
        cache_dir = "/scratch-shared/tpapandroeu/hf_cache"
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Using cache directory: {cache_dir}")
        
        # Get token from environment variable
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not hf_token:
            raise ValueError("HUGGING_FACE_TOKEN environment variable not set")
        
        # Initialize pipeline with debug=True
        pipeline = CLEF2025Pipeline(
            hf_token=hf_token,
            batch_size=16,
            cache_dir=cache_dir,
            debug=True
        )
        
        # Determine input/output paths based on task
        splits_dir = 'splits_snt' if task_num == '11' else 'splits_doc'
        input_file = f"data/CLEF2025/{splits_dir}/simpletext25_task{task_num}_test_part{part_num}.json"
        print(f"Loading test data from {input_file}...")
        with open(input_file, 'r') as f:
            test_data = json.load(f)
        
        # Prepare batches
        texts = [entry['complex'] for entry in test_data]
        print(f"Processing {len(texts)} entries from task {task_num}, part {part_num}...")
        print("Example text:", texts[0])
        
        # Process all texts in batches
        print("This will be done in two steps:")
        print("1. Detecting jargons in batches")
        print("2. Generating simplified text in batches")
        simplified_texts = pipeline.simplify_batch(texts)
        
        # Create results
        print("\nSaving results...")
        results = []
        output_prefix = 'Taiki_Task11_jargons' if task_num == '11' else 'Taiki_Task12_simplification'
        run_id = f'{output_prefix}_llama31_part{part_num}'
        
        for entry, simplified in zip(test_data, simplified_texts):
            result = entry.copy()
            result['run_id'] = run_id
            result['prediction'] = simplified
            results.append(result)
            print(f"Original: {entry['complex'][:100]}...")
            print(f"Simplified: {simplified[:100]}...")
            print("---")
        
        # Save results
        output_file = f"{output_prefix}_part{part_num}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()