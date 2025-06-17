"""
Implements evaluation metrics based on BERTScore
"""

import os
from typing import List
import torch
from bert_score import BERTScorer

# Set cache directory for HuggingFace models
os.environ['TRANSFORMERS_CACHE'] = '/scratch-shared/tpapandroeu/hf_cache'
os.environ['HF_HOME'] = '/scratch-shared/tpapandroeu/hf_cache'

# Initialize BERTScorer with RoBERTa model (done only once)
_scorer = None

def get_scorer():
    """Get or initialize the BERTScorer"""
    global _scorer
    if _scorer is None:
        _scorer = BERTScorer(
            model_type="roberta-large",
            num_layers=17,  # The layer that correlates best with human judgments
            batch_size=32,
            nthreads=4,
            all_layers=False,
            rescale_with_baseline=True,
            lang="en",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    return _scorer

def compute_bertscore(candidate: str, reference: str) -> float:
    """
    Compute BERTScore for a single sentence pair.
    Returns F1 score as a percentage (0-100).
    """
    if not candidate.strip() or not reference.strip():
        return 0.0
        
    scorer = get_scorer()
    P, R, F1 = scorer.score([candidate], [reference])
    # Ensure score is between 0 and 100
    return float(min(max(F1[0].item() * 100, 0.0), 100.0))

def corpus_bertscore(sys_sents: List[str], refs_sents: List[List[str]]) -> float:
    """
    Calculate corpus-level BERTScore.
    Args:
        sys_sents: List of system outputs
        refs_sents: List of lists of references
    Returns:
        Average F1 score as a percentage (0-100)
    """
    if not sys_sents or not refs_sents:
        return 0.0
        
    # For multiple references, we take the maximum score
    scores = []
    scorer = get_scorer()
    
    for sys_sent, refs in zip(sys_sents, refs_sents):
        if not isinstance(refs, list):
            refs = [refs]
            
        max_score = 0.0
        for ref in refs:
            P, R, F1 = scorer.score([sys_sent], [ref])
            # Ensure each score is between 0 and 100
            score = min(max(F1[0].item() * 100, 0.0), 100.0)
            max_score = max(max_score, float(score))
        scores.append(max_score)
    
    return sum(scores) / len(scores)  # Already in percentage 