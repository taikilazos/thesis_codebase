from typing import List, Tuple

# Examples showing different simplification strategies from PLABA
EXAMPLES = [
    # SUBSTITUTE example - keeping it very short and direct
    ("The patient exhibited tachycardia during examination",
     "The patient's heart was beating very fast during the checkup",
     "SUBSTITUTE: Replace technical terms with common words"),

    # EXPLAIN example - single clear sentence
    ("The MRI revealed a lesion",
     "The medical scan showed damaged tissue",
     "EXPLAIN: Add a brief clarification"),

    # GENERALIZE example - keeping it simple
    ("The patient requires ACE inhibitors for hypertension",
     "The patient needs blood pressure medicine",
     "GENERALIZE: Use simpler categories"),

    # OMIT example - very straightforward
    ("The lateral epicondylitis causes pain",
     "The condition causes pain",
     "OMIT: Remove technical details when possible"),

    # EXEMPLIFY example - clear and concise
    ("The patient shows signs of photophobia",
     "The patient is sensitive to bright light",
     "EXEMPLIFY: Give simple examples")
]

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

def create_few_shot_prompt(sentence: str) -> str:
    """Create a prompt with examples of different simplification strategies"""
    prompt = f"""{STRICT_INSTRUCTIONS}

Remember: Simplify language but keep ALL medical details accurate.
- Keep exact numbers
- Keep medical patterns (like 'myoclonic' if no exact simple equivalent exists)

Here are examples that keep precise medical meaning while using simpler language where possible:

"""
    # Add examples
    for orig, simp, _ in EXAMPLES:
        prompt += f"Medical: {orig}\nSimple: {simp}\n\n"

    # Add target sentence
    prompt += f"Now write one simplified sentence that keeps ALL medical details:\n{sentence}\n\nSimple:"
    return prompt

def create_combined_prompt(sentence: str, jargon_terms: List[str]) -> str:
    """Create a prompt that combines examples and jargon information"""
    prompt = f"""{STRICT_INSTRUCTIONS}

Remember: Simplify language but keep ALL medical details accurate.
- Keep exact numbers
- Keep medical patterns (like 'myoclonic' if no exact simple equivalent exists)

Here are examples that keep precise medical meaning while using simpler language where possible:

"""
    # Add examples
    for orig, simp, _ in EXAMPLES:
        prompt += f"Medical: {orig}\nSimple: {simp}\n\n"

    # Add jargon terms if any
    if jargon_terms:
        prompt += "Replace these terms ONLY if you can keep their exact medical meaning:\n"
        for term in jargon_terms:
            prompt += f"- {term}\n"
    
    # Add target sentence
    prompt += f"\nNow write one simplified sentence that keeps ALL medical details:\n{sentence}\n\nSimple:"
    return prompt

def get_prompt_function(prompt_type: str):
    """Get the appropriate prompt function based on type"""
    prompt_functions = {
        'simple': create_simple_prompt,
        'jargon': create_jargon_prompt,
        'few_shot': create_few_shot_prompt,
        'combined': create_combined_prompt
    }
    return prompt_functions.get(prompt_type) 