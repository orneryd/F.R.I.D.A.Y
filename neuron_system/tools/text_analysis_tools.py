"""
Text Analysis Tools for the neuron system.

Provides tools for analyzing text, counting letters, etc.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def count_letter_in_word(word: str, letter: str) -> int:
    """
    Count how many times a letter appears in a word.
    
    Args:
        word: The word to analyze
        letter: The letter to count (case-insensitive)
        
    Returns:
        Number of times the letter appears
    """
    word_lower = word.lower()
    letter_lower = letter.lower()
    count = word_lower.count(letter_lower)
    
    logger.info(f"Counted letter '{letter}' in word '{word}': {count} times")
    return count


def analyze_word(word: str) -> Dict[str, Any]:
    """
    Analyze a word and return various statistics.
    
    Args:
        word: The word to analyze
        
    Returns:
        Dictionary with analysis results
    """
    letter_counts = {}
    for letter in word.lower():
        if letter.isalpha():
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
    
    return {
        'word': word,
        'length': len(word),
        'letter_counts': letter_counts,
        'unique_letters': len(letter_counts),
        'vowels': sum(1 for c in word.lower() if c in 'aeiou'),
        'consonants': sum(1 for c in word.lower() if c.isalpha() and c not in 'aeiou')
    }
