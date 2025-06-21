from typing import List, Tuple

STRICT_INSTRUCTIONS = """IMPORTANT: Follow these rules exactly:
1. Write a clear sentence
2. Keep ALL medical distinctions and patterns
3. Keep exact numbers and measurements
4. Replace medical terms with plain words ONLY if meaning stays exactly the same
5. Keep medical terms if simplifying would lose precision
6. No explanations or notes
7. No multiple versions"""

def create_simple_prompt(sentence: str) -> str:
    """Create a basic simplification prompt"""
    return f"""{STRICT_INSTRUCTIONS}

Remember: Simplify language but keep ALL medical details accurate.
- Keep exact numbers
- Keep medical patterns (like 'myoclonic' if no exact simple equivalent exists)

Text to simplify: {sentence}

Write one simplified sentence:"""

def create_jargon_prompt(sentence: str, jargon_terms: List[str]) -> str:
    """Create a prompt that includes jargon information"""
    prompt = f"""{STRICT_INSTRUCTIONS}

Remember: Simplify language but keep ALL medical details accurate.
- Keep exact numbers
- Keep medical patterns (like 'myoclonic' if no exact simple equivalent exists)

Replace these terms ONLY if you can keep their exact medical meaning:
"""
    for term in jargon_terms:
        prompt += f"- {term}\n"
    
    prompt += f"\nText to simplify: {sentence}\n\nWrite one simplified sentence:"
    return prompt

def create_gt_jargons_prompt(sentence: str, jargon_terms: List[str]) -> str:
    """Prompt using ground truth jargons for the sentence."""
    prompt = f"""{STRICT_INSTRUCTIONS}

You must simplify the following terms in the sentence below (if present):
"""
    if jargon_terms:
        for term in jargon_terms:
            prompt += f"- {term}\n"
    else:
        prompt += "(No specific terms marked for simplification in this sentence.)\n"
    prompt += f"\nText to simplify: {sentence}\n\nWrite one simplified sentence:"
    return prompt

def create_gt_actions_prompt(sentence: str, jargon_action_pairs: List[Tuple[str, str, str]]) -> str:
    """Prompt using ground truth jargons and actions for the sentence."""
    prompt = f"""{STRICT_INSTRUCTIONS}

For each marked term in the sentence below, take the specified action:
"""
    if jargon_action_pairs:
        for jargon, action, replacement in jargon_action_pairs:
            if replacement:
                prompt += f"- {jargon}: {action} -> {replacement}\n"
            else:
                prompt += f"- {jargon}: {action}\n"
    else:
        prompt += "(No specific terms/actions marked for this sentence.)\n"
    prompt += f"\nText to simplify: {sentence}\n\nWrite one simplified sentence:"
    return prompt

def get_prompt_function(prompt_type: str):
    """Get the appropriate prompt function based on type"""
    prompt_functions = {
        'simple': create_simple_prompt,
        'jargon': create_jargon_prompt,
        'gt_jargons': create_gt_jargons_prompt,
        'gt_actions': create_gt_actions_prompt
    }
    return prompt_functions.get(prompt_type) 