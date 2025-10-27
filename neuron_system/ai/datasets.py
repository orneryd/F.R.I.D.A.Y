"""
Dataset Management - Dynamische Dataset-Loader.

Konsolidiert alle Dataset-Funktionalität:
- Built-in Datasets
- Custom Datasets
- Reddit Datasets
- File-based Datasets
"""

import logging
import json
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# DATASET REGISTRY
# ============================================================================

class DatasetRegistry:
    """Registry für verfügbare Datasets."""
    
    _datasets = {}
    
    @classmethod
    def register(cls, name: str, loader_func):
        """Register a dataset loader."""
        cls._datasets[name] = loader_func
        logger.debug(f"Registered dataset: {name}")
    
    @classmethod
    def get(cls, name: str):
        """Get dataset loader by name."""
        if name not in cls._datasets:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(cls._datasets.keys())}")
        return cls._datasets[name]
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available datasets."""
        return list(cls._datasets.keys())


# ============================================================================
# BUILT-IN DATASETS
# ============================================================================

def load_basic_ai_knowledge() -> Iterator[Dict[str, Any]]:
    """
    Basic AI knowledge dataset.
    
    Yields:
        Dict with 'question' and 'answer'
    """
    conversations = [
        {
            'question': 'What is AI?',
            'answer': 'Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction.'
        },
        {
            'question': 'What is machine learning?',
            'answer': 'Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.'
        },
        {
            'question': 'What is deep learning?',
            'answer': 'Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to learn hierarchical representations of data. It has been particularly successful in areas like image recognition and natural language processing.'
        },
        {
            'question': 'What is natural language processing?',
            'answer': 'Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. It bridges the gap between human communication and computer understanding.'
        },
        {
            'question': 'What is a neural network?',
            'answer': 'A neural network is a computing system inspired by biological neural networks. It consists of interconnected nodes (neurons) organized in layers that process information and learn patterns from data.'
        },
        {
            'question': 'What is supervised learning?',
            'answer': 'Supervised learning is a type of machine learning where the model is trained on labeled data. The algorithm learns to map inputs to outputs based on example input-output pairs.'
        },
        {
            'question': 'What is unsupervised learning?',
            'answer': 'Unsupervised learning is a type of machine learning where the model learns patterns from unlabeled data. It discovers hidden structures and relationships in the data without explicit guidance.'
        },
        {
            'question': 'What is reinforcement learning?',
            'answer': 'Reinforcement learning is a type of machine learning where an agent learns to make decisions by performing actions and receiving rewards or penalties. It learns through trial and error to maximize cumulative reward.'
        },
    ]
    
    for conv in conversations:
        yield conv


def load_programming_knowledge() -> Iterator[Dict[str, Any]]:
    """
    Programming knowledge dataset.
    
    Yields:
        Dict with 'question' and 'answer'
    """
    conversations = [
        {
            'question': 'What is Python?',
            'answer': 'Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms and has a comprehensive standard library.'
        },
        {
            'question': 'What is object-oriented programming?',
            'answer': 'Object-oriented programming (OOP) is a programming paradigm based on the concept of objects, which contain data and code. It emphasizes concepts like encapsulation, inheritance, and polymorphism.'
        },
        {
            'question': 'What is a function?',
            'answer': 'A function is a reusable block of code that performs a specific task. It can take inputs (parameters), process them, and return outputs. Functions help organize code and promote reusability.'
        },
        {
            'question': 'What is a variable?',
            'answer': 'A variable is a named storage location in memory that holds a value. Variables can store different types of data and their values can be changed during program execution.'
        },
    ]
    
    for conv in conversations:
        yield conv


def load_general_knowledge() -> Iterator[Dict[str, Any]]:
    """
    General knowledge dataset.
    
    Yields:
        Dict with 'question' and 'answer'
    """
    conversations = [
        {
            'question': 'What is science?',
            'answer': 'Science is a systematic enterprise that builds and organizes knowledge in the form of testable explanations and predictions about the universe. It uses observation, experimentation, and evidence-based reasoning.'
        },
        {
            'question': 'What is mathematics?',
            'answer': 'Mathematics is the study of numbers, quantities, shapes, and patterns. It provides a language and framework for describing and analyzing the world around us.'
        },
        {
            'question': 'What is technology?',
            'answer': 'Technology is the application of scientific knowledge for practical purposes. It includes tools, machines, techniques, and systems developed to solve problems and improve human life.'
        },
    ]
    
    for conv in conversations:
        yield conv


# Register built-in datasets
DatasetRegistry.register('basic-ai', load_basic_ai_knowledge)
DatasetRegistry.register('programming', load_programming_knowledge)
DatasetRegistry.register('general', load_general_knowledge)


# ============================================================================
# FILE-BASED DATASETS
# ============================================================================

def load_json_dataset(filepath: str) -> Iterator[Dict[str, Any]]:
    """
    Load dataset from JSON file.
    
    Expected format:
    [
        {"question": "...", "answer": "..."},
        {"question": "...", "answer": "..."}
    ]
    
    Or:
    {
        "conversations": [
            {"question": "...", "answer": "..."}
        ]
    }
    
    Args:
        filepath: Path to JSON file
        
    Yields:
        Dict with 'question' and 'answer'
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    logger.info(f"Loading dataset from: {filepath}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        conversations = data
    elif isinstance(data, dict) and 'conversations' in data:
        conversations = data['conversations']
    else:
        raise ValueError(f"Unknown JSON format in {filepath}")
    
    for conv in conversations:
        if 'question' in conv and 'answer' in conv:
            yield conv
        else:
            logger.warning(f"Skipping invalid conversation: {conv}")


def load_reddit_dataset(filepath: str) -> Iterator[Dict[str, Any]]:
    """
    Load Reddit-style dataset.
    
    Expected format:
    [
        {
            "title": "...",
            "selftext": "...",
            "comments": [
                {"body": "..."}
            ]
        }
    ]
    
    Args:
        filepath: Path to Reddit JSON file
        
    Yields:
        Dict with 'question' and 'answer'
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    logger.info(f"Loading Reddit dataset from: {filepath}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for post in data:
        title = post.get('title', '')
        selftext = post.get('selftext', '')
        comments = post.get('comments', [])
        
        # Create Q&A from title + selftext and top comment
        if title and comments:
            question = f"{title}\n{selftext}".strip()
            answer = comments[0].get('body', '') if comments else ''
            
            if question and answer and len(answer) > 10:
                yield {
                    'question': question,
                    'answer': answer
                }


# ============================================================================
# DATASET LOADER
# ============================================================================

class DatasetLoader:
    """
    Main dataset loader with support for multiple sources.
    """
    
    @staticmethod
    def load(source: str, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Load dataset from various sources.
        
        Args:
            source: Dataset source:
                - Built-in: 'basic-ai', 'programming', 'general'
                - File: path to .json file
                - Reddit: path to reddit .json file (use format='reddit')
            **kwargs: Additional arguments (e.g., format='reddit')
            
        Yields:
            Dict with 'question' and 'answer'
            
        Examples:
            # Built-in dataset
            loader = DatasetLoader.load('basic-ai')
            
            # JSON file
            loader = DatasetLoader.load('my_data.json')
            
            # Reddit file
            loader = DatasetLoader.load('reddit_data.json', format='reddit')
        """
        # Check if it's a built-in dataset
        if source in DatasetRegistry.list_available():
            logger.info(f"Loading built-in dataset: {source}")
            loader_func = DatasetRegistry.get(source)
            yield from loader_func()
            return
        
        # Check if it's a file
        path = Path(source)
        if path.exists():
            # Determine format
            format_type = kwargs.get('format', 'json')
            
            if format_type == 'reddit':
                yield from load_reddit_dataset(source)
            else:
                yield from load_json_dataset(source)
            return
        
        # Unknown source
        available = DatasetRegistry.list_available()
        raise ValueError(
            f"Unknown dataset source: {source}\n"
            f"Available built-in datasets: {available}\n"
            f"Or provide a path to a JSON file."
        )
    
    @staticmethod
    def list_available() -> Dict[str, str]:
        """
        List all available datasets.
        
        Returns:
            Dict of {name: description}
        """
        return {
            'basic-ai': 'Basic AI and ML knowledge (8 Q&A pairs)',
            'programming': 'Programming concepts (4 Q&A pairs)',
            'general': 'General knowledge (3 Q&A pairs)',
            'custom-file': 'Load from JSON file (provide path)',
            'reddit-file': 'Load from Reddit JSON (provide path with format="reddit")',
        }
    
    @staticmethod
    def print_available():
        """Print all available datasets."""
        datasets = DatasetLoader.list_available()
        
        print("\n" + "=" * 70)
        print("AVAILABLE DATASETS")
        print("=" * 70 + "\n")
        
        for name, description in datasets.items():
            print(f"  {name}")
            print(f"    {description}")
            print()
        
        print("=" * 70)
        print("Usage: python cli.py train --dataset DATASET_NAME")
        print("=" * 70 + "\n")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'DatasetLoader',
    'DatasetRegistry',
    'load_json_dataset',
    'load_reddit_dataset',
]
