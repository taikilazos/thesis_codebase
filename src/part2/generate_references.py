import json
import os
import nltk
from typing import Dict, List, Tuple, Set
from itertools import product
import argparse
from collections import defaultdict
import random
import pandas as pd
from tqdm import tqdm

def load_data(json_path: str) -> Dict:
    """Load data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_sentences_with_jargon(text: str, jargon_dict: Dict) -> List[Tuple[str, Dict]]:
    """
    Extract sentences that contain jargon terms and their corresponding jargon info
    Returns: List of (sentence, relevant_jargon_dict) tuples
    """
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    
    # For each sentence, find which jargon terms it contains
    sentence_jargon = []
    for sentence in sentences:
        sentence_dict = {}
        for term, actions in jargon_dict.items():
            if term in sentence:
                sentence_dict[term] = actions
        if sentence_dict:  # Only include sentences that contain jargon
            sentence_jargon.append((sentence, sentence_dict))
    
    return sentence_jargon

def get_term_options(actions: List) -> List[str]:
    """Get all possible substitutions for a term, including OMIT"""
    options = set()
    has_omit = False
    
    for action in actions:
        action_type, substitution = action
        if action_type == "OMIT":
            has_omit = True
        elif substitution:  # If there's a non-empty substitution text
            options.add(substitution)
    
    options = list(options)
    if has_omit:
        options.append("")  # Empty string represents OMIT
    
    return options

def generate_combinations(sentence: str, jargon_dict: Dict, max_combinations: int = None) -> List[str]:
    """
    Generate all possible combinations of simplifications for a sentence.
    If max_combinations is specified, randomly sample that many combinations.
    """
    # Get all terms and their possible substitutions
    terms = []
    substitutions = []
    
    for term, actions in jargon_dict.items():
        if term in sentence:
            terms.append(term)
            substitutions.append(get_term_options(actions))
    
    # Calculate total possible combinations
    total = 1
    for subs in substitutions:
        total *= len(subs)
    
    # Generate combinations
    if max_combinations and total > max_combinations:
        # Random sampling approach
        simplified_sentences = set()
        attempts = 0
        max_attempts = max_combinations * 2  # Allow some extra attempts for duplicates
        
        while len(simplified_sentences) < max_combinations and attempts < max_attempts:
            # Generate one random combination
            current = sentence
            for term, options in zip(terms, substitutions):
                replacement = random.choice(options)
                if replacement:  # If not OMIT
                    current = current.replace(term, replacement)
                else:  # OMIT case
                    current = current.replace(term, "")
            simplified_sentences.add(current)
            attempts += 1
        
        return list(simplified_sentences)
    else:
        # Generate all combinations
        all_combinations = []
        for combination in product(*substitutions):
            current = sentence
            for term, replacement in zip(terms, combination):
                if replacement:  # If not OMIT
                    current = current.replace(term, replacement)
                else:  # OMIT case
                    current = current.replace(term, "")
            all_combinations.append(current)
        return all_combinations

def analyze_dataset_size(data_dir: str, json_path: str, mode: str = 'full', threshold: int = 100):
    """Analyze potential dataset size for sentence-level simplification"""
    # Load data
    data = load_data(json_path)
    print(f"\nAnalyzing data from {json_path}")
    print(f"Mode: {'Full' if mode == 'full' else f'Thresholded (max {threshold} combinations)'}")
    print(f"Number of documents: {len(data)}")
    
    # Statistics
    total_sentences = 0
    sentences_with_jargon = 0
    total_combinations = 0
    jargon_per_sentence = defaultdict(int)
    combinations_distribution = defaultdict(int)
    
    # Process each document
    for doc_id, jargon_dict in data.items():
        # Read original text
        src_path = os.path.join(data_dir, "abstracts", f"{doc_id}.src.txt")
        if not os.path.exists(src_path):
            print(f"Warning: Source file not found for {doc_id}")
            continue
            
        with open(src_path, 'r') as f:
            text = f.read().strip()
        
        # Get sentences containing jargon
        sentences = nltk.sent_tokenize(text)
        total_sentences += len(sentences)
        
        sentence_jargon = get_sentences_with_jargon(text, jargon_dict)
        sentences_with_jargon += len(sentence_jargon)
        
        # Analyze each sentence
        for sentence, sent_jargon in sentence_jargon:
            # Count jargon terms
            num_jargon = len(sent_jargon)
            jargon_per_sentence[num_jargon] += 1
            
            # Generate combinations
            combinations = generate_combinations(
                sentence, 
                sent_jargon,
                max_combinations=threshold if mode == 'threshold' else None
            )
            
            num_combinations = len(combinations)
            total_combinations += num_combinations
            combinations_distribution[num_combinations] += 1
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total sentences: {total_sentences}")
    print(f"Sentences containing jargon: {sentences_with_jargon} ({sentences_with_jargon/total_sentences*100:.1f}%)")
    print(f"Total possible combinations: {total_combinations:,}")
    print(f"Average combinations per sentence: {total_combinations/sentences_with_jargon:.1f}")
    
    print("\nJargon terms per sentence distribution:")
    for num_terms in sorted(jargon_per_sentence.keys()):
        count = jargon_per_sentence[num_terms]
        print(f"{num_terms} terms: {count} sentences ({count/sentences_with_jargon*100:.1f}%)")
    
    print("\nCombinations per sentence distribution:")
    for combs in sorted(combinations_distribution.keys()):
        count = combinations_distribution[combs]
        print(f"{combs} combinations: {count} sentences ({count/sentences_with_jargon*100:.1f}%)")

def generate_dataset(data_dir: str, json_path: str, output_dir: str, mode: str = 'full', threshold: int = 100):
    """Generate and save the dataset of sentence pairs"""
    # Load data
    data = load_data(json_path)
    print(f"\nGenerating dataset from {json_path}")
    print(f"Mode: {'Full' if mode == 'full' else f'Thresholded (max {threshold} combinations)'}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for DataFrame
    dataset_rows = []
    total_combinations = 0
    
    # Process each document with progress bar
    for doc_id in tqdm(data.keys(), desc="Processing documents"):
        jargon_dict = data[doc_id]
        
        # Read original text
        src_path = os.path.join(data_dir, "abstracts", f"{doc_id}.src.txt")
        if not os.path.exists(src_path):
            print(f"Warning: Source file not found for {doc_id}")
            continue
            
        with open(src_path, 'r') as f:
            text = f.read().strip()
        
        # Get sentences containing jargon
        sentence_jargon = get_sentences_with_jargon(text, jargon_dict)
        
        # Generate combinations for each sentence
        for sentence_idx, (sentence, sent_jargon) in enumerate(sentence_jargon):
            combinations = generate_combinations(
                sentence, 
                sent_jargon,
                max_combinations=threshold if mode == 'threshold' else None
            )
            
            # Add to dataset
            for comb_idx, simplified in enumerate(combinations):
                row = {
                    'doc_id': doc_id,
                    'sentence_idx': sentence_idx,
                    'combination_idx': comb_idx,
                    'original_sentence': sentence,
                    'simplified_sentence': simplified,
                    'num_jargon_terms': len(sent_jargon),  # Added this field for analysis
                }
                dataset_rows.append(row)
            total_combinations += len(combinations)
    
    # Save statistics
    stats = {
        'total_documents': len(data),
        'total_combinations': total_combinations,
        'total_unique_sentences': len(set(row['original_sentence'] for row in dataset_rows)),
        'average_combinations_per_sentence': total_combinations / len(set(row['original_sentence'] for row in dataset_rows)),
        'mode': mode,
        'threshold': threshold if mode == 'threshold' else None
    }
    
    # Save as Parquet
    df = pd.DataFrame(dataset_rows)
    parquet_path = os.path.join(output_dir, f"simplification_pairs_{mode}.parquet")
    df.to_parquet(parquet_path, index=False)
    
    # Save as JSON (in a readable format)
    json_path = os.path.join(output_dir, f"simplification_pairs_{mode}.json")
    json_data = {
        'metadata': stats,
        'pairs': dataset_rows
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Save statistics separately
    stats_path = os.path.join(output_dir, f"dataset_stats_{mode}.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset generated successfully!")
    print(f"Total combinations: {total_combinations:,}")
    print(f"Outputs saved to:")
    print(f"- Parquet: {parquet_path}")
    print(f"- JSON: {json_path}")
    print(f"- Stats: {stats_path}")
    
    # Print sample for verification
    print("\nSample entries:")
    for i, row in enumerate(dataset_rows[:2]):  # Show first 2 entries
        print(f"\nEntry {i+1}:")
        print(f"Original: {row['original_sentence']}")
        print(f"Simplified: {row['simplified_sentence']}")
        print(f"Doc ID: {row['doc_id']}, Sentence: {row['sentence_idx']}, Combination: {row['combination_idx']}")

def main():
    parser = argparse.ArgumentParser(description='Generate sentence-level simplification dataset')
    parser.add_argument('--data_dir', type=str, default="data/PLABA_2024-Task_1",
                      help='Directory containing the data')
    parser.add_argument('--json_path', type=str, default="data/PLABA_2024-Task_1/task_1_testing.json",
                      help='Path to the JSON file to analyze')
    parser.add_argument('--output_dir', type=str, default="data/PLABA_2024-Task_1/sentence_pairs",
                      help='Directory to save the generated dataset')
    parser.add_argument('--mode', type=str, choices=['full', 'threshold'], default='full',
                      help='Whether to generate all combinations or use thresholding')
    parser.add_argument('--threshold', type=int, default=100,
                      help='Maximum number of combinations per sentence when using threshold mode')
    
    args = parser.parse_args()
    generate_dataset(args.data_dir, args.json_path, args.output_dir, args.mode, args.threshold)

if __name__ == "__main__":
    main() 