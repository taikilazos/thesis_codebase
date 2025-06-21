import argparse
import json
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import nltk
from datetime import datetime
import numpy as np
import re

# Import local metrics
from easse_metrics.bleu import corpus_bleu
from easse_metrics.fkgl import corpus_fkgl
from easse_metrics.sari import corpus_sari
from easse_metrics.bertscore import corpus_bertscore

from models import get_model
from prompts import get_prompt_function


class JargonDetector:
    def __init__(self, model_path: str = "output/plaba/roberta_1a.pt"):
        """Initialize the binary jargon detection model (PLABA 1a)"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large", add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained("roberta-large", num_labels=2)
        
        # Load model weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping for binary classification
        self.id2label = {
            0: 'O',
            1: 'JARGON'
        }
        
        nltk.download('punkt', quiet=True)
    
    def detect_jargons(self, text: str) -> dict:
        """Detect jargons in a text using sliding windows for long sentences (binary version)"""
        text_jargons = {'JARGON': set()}
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            if not tokens:
                continue
            all_subtokens = self.tokenizer.tokenize(" ".join(tokens))
            WINDOW_SIZE = 200
            OVERLAP = 50
            if len(all_subtokens) > 250:
                for start_idx in range(0, len(tokens), WINDOW_SIZE - OVERLAP):
                    window_tokens = tokens[start_idx:start_idx + WINDOW_SIZE]
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
                encoding = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    padding=True,
                    truncation=True,
                    max_length=250,
                    return_tensors='pt'
                )
                self._process_window(encoding, tokens, text_jargons)
        # Minimal deduplication: remove duplicate words in each jargon phrase
        def dedup(phrase):
            out = []
            for w in phrase.split():
                if w not in out:
                    out.append(w)
            return ' '.join(out)
        return {k: [dedup(j) for j in v] for k, v in text_jargons.items()}
    
    def _process_window(self, encoding, tokens, text_jargons):
        """Helper method to process a window of tokens and update jargons (binary version)"""
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
        
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
                else:
                    if current_term:
                        text_jargons['JARGON'].add(' '.join(current_term))
                        current_term = []
        
        if current_term:
            text_jargons['JARGON'].add(' '.join(current_term))

def load_submission_ids(submission_path):
    with open(submission_path, 'r') as f:
        data = json.load(f)
    return list(data.keys())

def load_document(path):
    with open(path, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
    return lines

def find_jargons_in_sentence(sentence: str, gt_jargons_dict: Dict[str, List[List[str]]]) -> List[str]:
    """Return list of jargons from gt_jargons_dict that appear in the sentence (case-insensitive, whole word)."""
    found = []
    for jargon in gt_jargons_dict.keys():
        # Use regex for whole word, case-insensitive
        if re.search(rf'\b{re.escape(jargon)}\b', sentence, re.IGNORECASE):
            found.append(jargon)
    return found

def find_jargons_actions_in_sentence(sentence: str, gt_jargons_dict: Dict[str, List[List[str]]]) -> List[Tuple[str, str, str]]:
    """Return list of (jargon, action, replacement/explanation) for jargons in sentence (first action only)."""
    found = []
    for jargon, actions in gt_jargons_dict.items():
        if re.search(rf'\b{re.escape(jargon)}\b', sentence, re.IGNORECASE):
            if actions:
                action, replacement = actions[0][0], actions[0][1]
                found.append((jargon, action, replacement))
    return found

def simplify_document(lines, model, prompt_func, jargon_detector=None, gt_jargons_dict=None, prompt_type=None):
    simplified_lines = []
    prompts = []
    outputs = []
    for line in lines:
        if not line.strip():
            simplified_lines.append("")
            prompts.append("")
            outputs.append("")
            continue
        if prompt_type == 'gt_jargons' and gt_jargons_dict is not None:
            matched_jargons = find_jargons_in_sentence(line, gt_jargons_dict)
            prompt = prompt_func(line, matched_jargons)
        elif prompt_type == 'gt_actions' and gt_jargons_dict is not None:
            matched_jargons_actions = find_jargons_actions_in_sentence(line, gt_jargons_dict)
            prompt = prompt_func(line, matched_jargons_actions)
        elif jargon_detector and prompt_func.__name__ == 'create_jargon_prompt':
            jargons = jargon_detector.detect_jargons(line)
            jargon_terms = jargons.get('JARGON', [])
            prompt = prompt_func(line, jargon_terms)
        else:
            prompt = prompt_func(line)
        simplified = model.simplify(prompt)
        simplified_lines.append(simplified)
        prompts.append(prompt)
        outputs.append(simplified)
    return simplified_lines, prompts, outputs

def main():
    parser = argparse.ArgumentParser(description='PLABA Task 1 Abstract Simplification')
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to PLABA_2024-Task_1 directory')
    parser.add_argument('--reference_system', type=str, required=True, help='Which NLM reference to use (e.g., NLM_1)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--max_abstracts', type=int, default=None, help='Maximum number of abstracts to process')
    parser.add_argument('--prompt_type', type=str, choices=['simple', 'jargon', 'gt_jargons', 'gt_actions'], default='simple', help='Type of prompt to use')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='Model to use')
    parser.add_argument('--cache_dir', type=str, default="/scratch-shared/tpapandroeu/hf_cache", help='Model cache dir')
    args = parser.parse_args()

    # Print arguments
    print("=== ARGUMENTS ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("================")

    os.makedirs(args.output_dir, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(args.output_dir, f"results_{args.reference_system}_{date_str}.json")

    # Load abstract IDs
    submission_path = os.path.join(args.dataset_root, 'submission_template.json')
    abstract_ids = load_submission_ids(submission_path)
    if args.max_abstracts is not None:
        abstract_ids = abstract_ids[:args.max_abstracts]

    # Load ground truth jargons/actions if needed
    gt_jargons_data = None
    if args.prompt_type in ['gt_jargons', 'gt_actions']:
        gt_path = os.path.join('data', 'PLABA_2024-Task_1', 'task_1_testing.json')
        with open(gt_path, 'r') as f:
            gt_jargons_data = json.load(f)

    # Initialize model and prompt
    model = get_model(args.model_name, args.cache_dir)
    prompt_func = get_prompt_function(args.prompt_type)
    jargon_detector = None
    if prompt_func and prompt_func.__name__ == 'create_jargon_prompt':
        jargon_detector = JargonDetector()
    if not prompt_func:
        raise ValueError(f"Invalid prompt type: {args.prompt_type}")

    results = []
    for abs_id in tqdm(abstract_ids, desc="Processing abstracts"):
        # Load original document
        src_path = os.path.join(args.dataset_root, 'abstracts', f'{abs_id}.src.txt')
        if not os.path.exists(src_path):
            print(f"Warning: {src_path} not found, skipping.")
            continue
        original_lines = load_document(src_path)
        original_doc = "\n".join(original_lines)

        # Prepare ground truth jargons/actions for this abstract if needed
        gt_jargons_dict = None
        if gt_jargons_data is not None and abs_id in gt_jargons_data:
            gt_jargons_dict = gt_jargons_data[abs_id]

        # Simplify each line
        simplified_lines, prompts, outputs = simplify_document(
            original_lines, model, prompt_func, jargon_detector,
            gt_jargons_dict=gt_jargons_dict, prompt_type=args.prompt_type
        )
        simplified_doc = "\n".join(simplified_lines)

        # Load reference document
        ref_path = os.path.join(args.dataset_root, 'references', args.reference_system, f'{abs_id}.tgt.txt')
        if not os.path.exists(ref_path):
            print(f"Warning: {ref_path} not found, skipping.")
            continue
        reference_lines = load_document(ref_path)
        reference_doc = "\n".join(reference_lines)

        # Compute metrics (document level)
        sari = corpus_sari([original_doc], [simplified_doc], [[reference_doc]])
        bleu = corpus_bleu([simplified_doc], [[reference_doc]])
        fkgl = corpus_fkgl([simplified_doc])
        bertscore_f1 = corpus_bertscore([simplified_doc], [[reference_doc]])

        results.append({
            'id': abs_id,
            'original': original_doc,
            'simplified': simplified_doc,
            'reference': reference_doc,
            'prompts': prompts,
            'outputs': outputs,
            'metrics': {
                'SARI': sari,
                'BLEU': bleu,
                'FKGL': fkgl,
                'BERTScore': bertscore_f1
            }
        })

    # After your results loop
    sari_scores = [r['metrics']['SARI'] for r in results]
    bleu_scores = [r['metrics']['BLEU'] for r in results]
    fkgl_scores = [r['metrics']['FKGL'] for r in results]
    bertscore_scores = [r['metrics']['BERTScore'] for r in results]

    print("\n=== AVERAGE METRICS OVER ALL ABSTRACTS ===")
    print(f"Average SARI: {np.mean(sari_scores):.2f}")
    print(f"Average BLEU: {np.mean(bleu_scores):.2f}")
    print(f"Average FKGL: {np.mean(fkgl_scores):.2f}")
    print(f"Average BERTScore: {np.mean(bertscore_scores):.2f}")

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main() 