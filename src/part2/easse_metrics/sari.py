"""
Implements evaluation metrics based on SARI score
"""

from collections import Counter
from typing import List, Set, Dict, Tuple

def extract_ngrams(text: str, n: int = None) -> Counter:
    """Extract n-grams from text. If n is None, extract 1-4 grams."""
    tokens = text.lower().split()
    ngrams = Counter()
    
    if not tokens:
        return ngrams
    
    # If n is specified, only extract that size
    if n is not None:
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i + n])
            ngrams[ngram] += 1
        return ngrams
    
    # Otherwise extract 1-4 grams
    for n in range(1, 5):  # 1-4 grams
        if len(tokens) >= n:
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i + n])
                ngrams[ngram] += 1
    
    return ngrams

def get_ngram_ops(orig_tokens: Set[str], sys_tokens: Set[str], ref_tokens: Set[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    """Get the sets of n-grams for add, delete and keep operations."""
    add_ops = ref_tokens - orig_tokens  # Added by reference
    keep_ops = orig_tokens & ref_tokens  # Should be kept
    del_ops = orig_tokens - ref_tokens   # Should be deleted
    return add_ops, keep_ops, del_ops

def compute_precision_recall_f1(sys_ngrams: Set[str], ref_ngrams: Set[str]) -> Tuple[float, float, float]:
    """Compute precision, recall and F1 for n-gram sets."""
    if not sys_ngrams or not ref_ngrams:
        return 0.0, 0.0, 0.0
    
    # Find matches
    matches = sys_ngrams & ref_ngrams
    
    # Calculate metrics
    precision = len(matches) / len(sys_ngrams) if sys_ngrams else 0.0
    recall = len(matches) / len(ref_ngrams) if ref_ngrams else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def compute_sari(orig: str, sys: str, ref: str) -> float:
    """
    Compute SARI score for a single sentence.
    SARI = (F1_add + F1_keep + P_del) / 3
    """
    if not orig.strip() or not sys.strip() or not ref.strip():
        return 0.0
    
    # Get n-grams (1-4) for each text
    orig_ngrams = set(extract_ngrams(orig))
    sys_ngrams = set(extract_ngrams(sys))
    ref_ngrams = set(extract_ngrams(ref))
    
    # Get operation sets
    add_refs, keep_refs, del_refs = get_ngram_ops(orig_ngrams, sys_ngrams, ref_ngrams)
    
    # Calculate scores for each operation
    # Add: n-grams in sys and ref but not in orig
    sys_add = sys_ngrams - orig_ngrams
    _, _, add_f1 = compute_precision_recall_f1(sys_add, add_refs)
    
    # Keep: n-grams in both orig and ref that are kept in sys
    sys_keep = sys_ngrams & orig_ngrams
    _, _, keep_f1 = compute_precision_recall_f1(sys_keep, keep_refs)
    
    # Delete: n-grams in orig that should be deleted (not in ref)
    sys_del = orig_ngrams - sys_ngrams
    del_p, _, _ = compute_precision_recall_f1(sys_del, del_refs)
    
    # Calculate final SARI score
    sari = (add_f1 + keep_f1 + del_p) / 3.0
    return sari * 100  # Convert to percentage

def corpus_sari(orig_sents: List[str], sys_sents: List[str], refs_sents: List[List[str]]) -> float:
    """
    Calculate corpus-level SARI score.
    Args:
        orig_sents: List of original sentences
        sys_sents: List of system outputs
        refs_sents: List of lists of references
    """
    if not sys_sents or not refs_sents or not orig_sents:
        return 0.0
    
    # Calculate SARI for each sentence and average
    scores = []
    for orig, sys, refs in zip(orig_sents, sys_sents, refs_sents):
        # For multiple references, take the maximum SARI score
        max_score = 0.0
        for ref in refs:
            score = compute_sari(orig, sys, ref)
            max_score = max(max_score, score)
        scores.append(max_score)
    
    return sum(scores) / len(scores)  # Already in percentage 