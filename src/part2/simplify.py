import argparse
import json
import os
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import nltk

# Import local metrics
from easse_metrics.sari import corpus_sari, compute_sari
from easse_metrics.bertscore import compute_bertscore

from models import get_model
from prompts import get_prompt_function

class JargonDetector:
    def __init__(self, model_path: str = "/home/tpapandroeu/reproducibility/output/medreadme/roberta_base_7-cls.pt"):
        """Initialize the jargon detection model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained("roberta-base", num_labels=8)
        
        # Load model weights
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping
        self.id2label = {
            0: 'O', 1: 'GOOGLE_EASY', 2: 'GOOGLE_HARD',
            3: 'MEDICAL_NAME', 4: 'MEDICAL_ABBR',
            5: 'GENERAL_ABBR', 6: 'GENERAL_COMPLEX', 7: 'OTHER'
        }
        
        nltk.download('punkt', quiet=True)
    
    def detect_jargons(self, text: str) -> Dict[str, List[str]]:
        """Detect jargons in a text using sliding windows for long sentences"""
        # Initialize jargons dict with sets for deduplication
        text_jargons = {
            'MEDICAL_NAME': set(), 'MEDICAL_ABBR': set(),
            'GENERAL_COMPLEX': set(), 'GOOGLE_HARD': set(),
            'GENERAL_ABBR': set(), 'GOOGLE_EASY': set(),
            'OTHER': set()
        }
        
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
            all_subtokens = self.tokenizer.tokenize(" ".join(tokens))
            
            # If sentence exceeds token limit, process it in windows
            WINDOW_SIZE = 200  # Smaller than model limit of 250
            OVERLAP = 50
            
            if len(all_subtokens) > 250:
                # Process sentence in overlapping windows
                for start_idx in range(0, len(tokens), WINDOW_SIZE - OVERLAP):
                    window_tokens = tokens[start_idx:start_idx + WINDOW_SIZE]
                    
                    # Process window
                    encoding = self.tokenizer(
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
                encoding = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    padding=True,
                    truncation=True,
                    max_length=250,
                    return_tensors='pt'
                )
                
                self._process_window(encoding, tokens, text_jargons)
        
        # Convert sets to lists
        return {k: list(v) for k, v in text_jargons.items()}
    
    def _process_window(self, encoding, tokens, text_jargons):
        """Helper method to process a window of tokens and update jargons"""
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Process predictions
        current_term = []
        current_label = None
        word_ids = encoding.word_ids(batch_index=0)
        
        for idx, (pred, word_id) in enumerate(zip(predictions[0], word_ids)):
            if word_id is not None:
                label = self.id2label[pred.item()]
                if label != 'O':
                    token = tokens[word_id]
                    if not current_term:
                        current_term = [token]
                        current_label = label
                    elif label == current_label:
                        current_term.append(token)
                    else:
                        if current_term:
                            text_jargons[current_label].add(' '.join(current_term))
                        current_term = [token]
                        current_label = label
                else:
                    if current_term:
                        text_jargons[current_label].add(' '.join(current_term))
                        current_term = []
                        current_label = None
        
        if current_term:
            text_jargons[current_label].add(' '.join(current_term))

def load_dataset(parquet_path: str, debug: bool = False) -> pd.DataFrame:
    """Load the dataset from parquet file"""
    df = pd.read_parquet(parquet_path)
    if debug:
        print(f"\nDataset sample ({min(3, len(df))} rows):")
        print(df[['doc_id', 'original_sentence', 'simplified_sentence']].head(3))
        print(f"\nTotal rows: {len(df)}")
    return df

def calculate_sentence_sari(orig: str, sys: str, ref: str) -> float:
    """Calculate SARI score for a single sentence"""
    # We use corpus_sari but with single sentences
    return corpus_sari([orig], [sys], [[ref]])

def calculate_sentence_bertscore(sys: str, ref: str) -> float:
    """Calculate BERTScore for a single sentence"""
    return compute_bertscore(sys, ref)  # Already returns percentage

def calculate_metrics(original: str, simplified: str, reference: str, debug: bool = False) -> Dict[str, float]:
    """Calculate SARI and BERTScore metrics for a sentence pair"""
    if debug:
        print("\nCalculating metrics for:")
        print(f"Original: {original}")
        print(f"Simplified: {simplified}")
        print(f"Reference: {reference}")
    
    # Calculate metrics
    sari = calculate_sentence_sari(original, simplified, reference)
    bertscore = calculate_sentence_bertscore(simplified, reference)
    
    if debug:
        print(f"\nSARI score: {sari:.2f}")
        print(f"BERTScore: {bertscore:.2f}")
    
    return {
        'sari': sari,
        'bertscore': bertscore
    }

def process_unique_sentences(
    dataset: pd.DataFrame,
    model,
    prompt_func,
    output_file: str,
    debug: bool = False,
    max_samples: int = None
) -> None:
    """Process unique sentences and save model outputs"""
    # Get unique sentences by doc_id and sentence_idx
    unique_df = dataset.drop_duplicates(subset=['doc_id', 'sentence_idx', 'original_sentence'])
    if debug:
        print(f"\nTotal unique sentences: {len(unique_df)}")
        unique_df = unique_df.head(5)  # Take first 5 for debug
        print("Debug mode: using first 5 unique sentences")
    elif max_samples:
        unique_df = unique_df.head(max_samples)
    
    # Initialize jargon detector
    jargon_detector = JargonDetector()
    
    # Generate simplifications
    results = []
    for _, row in tqdm(unique_df.iterrows(), total=len(unique_df), desc="Generating simplifications"):
        # Detect jargons
        jargons = jargon_detector.detect_jargons(row['original_sentence'])
        
        # Process and deduplicate jargon terms
        all_terms = set()  # Use set for deduplication
        for category, terms in jargons.items():
            if terms and category != 'OTHER':
                # Process each term
                for term in terms:
                    # Split term into words
                    words = term.split()
                    # For single words or unique multi-word terms, add directly
                    if len(words) == 1 or term not in all_terms:
                        all_terms.add(term)
        
        # Sort terms by length (longer terms first) and then alphabetically
        sorted_terms = sorted(all_terms, key=lambda x: (-len(x), x.lower()))
        
        # Create prompt
        if prompt_func.__name__ in ['create_jargon_prompt', 'create_combined_prompt']:
            prompt = prompt_func(row['original_sentence'], sorted_terms)
        else:
            prompt = prompt_func(row['original_sentence'])
        
        # Get model output
        simplified = model.simplify(prompt, prompt)
        
        # Store result
        result = {
            'doc_id': row['doc_id'],
            'sentence_idx': int(row['sentence_idx']),
            'original': row['original_sentence'],
            'simplified': simplified,
            'jargon_terms': sorted_terms  # Store detected jargon terms
        }
        results.append(result)
        
        if debug:
            print(f"\nProcessed sentence {len(results)}:")
            print(f"Original: {row['original_sentence']}")
            print(f"Detected jargons: {sorted_terms}")
            print(f"Simplified: {simplified}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    if debug:
        print(f"\nSaved {len(results)} results to {output_file}")

def calculate_all_metrics(
    dataset: pd.DataFrame,
    simplified_file: str,
    output_file: str,
    metrics_file: str,
    debug: bool = False
) -> None:
    """Calculate metrics for all references using simplified outputs"""
    # Load simplified outputs
    with open(simplified_file, 'r') as f:
        simplified_outputs = json.load(f)
    
    # Create lookup dictionary
    simplified_dict = {(item['doc_id'], item['sentence_idx']): item['simplified'] 
                      for item in simplified_outputs}
    
    # Process all references and calculate metrics
    results = []
    all_metrics = {'sari': [], 'bertscore': []}
    
    # Group by doc_id and sentence_idx to process all references together
    grouped = dataset.groupby(['doc_id', 'sentence_idx'])
    
    for (doc_id, sent_idx), group in tqdm(grouped, desc="Calculating metrics"):
        # Get simplified output
        simplified = simplified_dict.get((doc_id, sent_idx))
        if not simplified:
            continue
        
        # Calculate metrics for all references
        reference_metrics = []
        for _, row in group.iterrows():
            metrics = calculate_metrics(
                row['original_sentence'],
                simplified,
                row['simplified_sentence'],
                debug=debug
            )
            reference_metrics.append(metrics)
        
        # Average metrics across references
        avg_metrics = {
            'sari': float(np.mean([m['sari'] for m in reference_metrics])),
            'bertscore': float(np.mean([m['bertscore'] for m in reference_metrics]))
        }
        
        # Store metrics for overall average
        all_metrics['sari'].append(avg_metrics['sari'])
        all_metrics['bertscore'].append(avg_metrics['bertscore'])
        
        # Store result with first reference (for example)
        first_ref = group.iloc[0]
        result = {
            'doc_id': doc_id,
            'sentence_idx': int(sent_idx),
            'original': first_ref['original_sentence'],
            'simplified': simplified,
            'reference': first_ref['simplified_sentence'],
            'metrics': avg_metrics
        }
        results.append(result)
        
        if debug:
            print(f"\nProcessed {doc_id} sentence {sent_idx}:")
            print(f"Metrics (averaged over {len(group)} references): {avg_metrics}")
    
    # Save detailed results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate and save overall metrics
    overall_metrics = {
        'sari': {
            'mean': float(np.mean(all_metrics['sari'])),
            'std': float(np.std(all_metrics['sari']))
        },
        'bertscore': {
            'mean': float(np.mean(all_metrics['bertscore'])),
            'std': float(np.std(all_metrics['bertscore']))
        }
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    if debug:
        print(f"\nSaved {len(results)} results to {output_file}")
        print(f"Overall metrics saved to {metrics_file}")
        print("Final metrics:", overall_metrics)

def main():
    parser = argparse.ArgumentParser(description='Run text simplification inference')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the parquet dataset')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                      help='Name of the model to use')
    parser.add_argument('--prompt_type', type=str, choices=['simple', 'jargon', 'few_shot', 'combined'],
                      default='combined', help='Type of prompt to use')
    parser.add_argument('--cache_dir', type=str, default="/scratch-shared/tpapandroeu/hf_cache",
                      help='Directory for caching models')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save outputs')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug prints')
    parser.add_argument('--phase', type=int, choices=[1, 2], required=True,
                      help='Phase to run: 1 for generation, 2 for metrics')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup output files
    model_name = args.model_name.split('/')[-1]
    base_output = os.path.join(args.output_dir, f"{model_name}_{args.prompt_type}")
    simplified_file = f"{base_output}_simplified.json"
    output_file = f"{base_output}.json"
    metrics_file = f"{base_output}_metrics.json"
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    dataset = load_dataset(args.data_path, debug=args.debug)
    
    if args.phase == 1:
        # Initialize model
        print(f"Initializing {args.model_name}")
        model = get_model(args.model_name, args.cache_dir)
        
        # Get prompt function
        prompt_func = get_prompt_function(args.prompt_type)
        if not prompt_func:
            raise ValueError(f"Invalid prompt type: {args.prompt_type}")
        
        print(f"Processing unique sentences with {args.prompt_type} prompts")
        process_unique_sentences(
            dataset=dataset,
            model=model,
            prompt_func=prompt_func,
            output_file=simplified_file,
            debug=args.debug
        )
        print(f"Phase 1 complete. Results saved to {simplified_file}")
        
    elif args.phase == 2:
        print("Calculating metrics")
        calculate_all_metrics(
            dataset=dataset,
            simplified_file=simplified_file,
            output_file=output_file,
            metrics_file=metrics_file,
            debug=args.debug
        )
        print(f"Phase 2 complete. Results saved to {output_file}")
        print(f"Overall metrics saved to {metrics_file}")

if __name__ == "__main__":
    main() 