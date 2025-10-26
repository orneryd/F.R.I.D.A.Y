"""
Reddit Conversation Dataset Loader.

Loads conversations from Reddit dataset with quality filtering.
"""

import json
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class RedditLoader:
    """
    Loader for Reddit conversation dataset.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize Reddit loader.
        
        Args:
            data_path: Path to Reddit dataset directory
        """
        self.data_path = Path(data_path) if data_path else None
        self.conversations_loaded = 0
    
    def load_from_file(
        self,
        file_path: str,
        max_conversations: Optional[int] = None,
        min_score: int = 2
    ) -> List[Tuple[str, str]]:
        """
        Load conversations from a Reddit JSON file.
        
        Args:
            file_path: Path to JSON file
            max_conversations: Maximum number to load (None = all)
            min_score: Minimum Reddit score (upvotes - downvotes)
            
        Returns:
            List of (question, answer) tuples
        """
        conversations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if max_conversations and len(conversations) >= max_conversations:
                        break
                    
                    try:
                        data = json.loads(line.strip())
                        
                        # Extract conversation
                        question, answer = self._extract_conversation(data, min_score)
                        
                        if question and answer:
                            conversations.append((question, answer))
                            self.conversations_loaded += 1
                    
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON at line {line_num}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        continue
            
            logger.info(f"Loaded {len(conversations)} conversations from {file_path}")
            return conversations
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            return []
    
    def _extract_conversation(
        self,
        data: Dict[str, Any],
        min_score: int
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract question-answer pair from Reddit data.
        
        Args:
            data: Reddit post/comment data
            min_score: Minimum score threshold
            
        Returns:
            Tuple of (question, answer) or (None, None)
        """
        try:
            # Reddit format varies, try different structures
            
            # Format 1: Direct question-answer
            if 'question' in data and 'answer' in data:
                question = data['question'].strip()
                answer = data['answer'].strip()
                score = data.get('score', 0)
                
                if score >= min_score:
                    return question, answer
            
            # Format 2: Parent-child comments
            elif 'parent' in data and 'response' in data:
                question = data['parent'].strip()
                answer = data['response'].strip()
                score = data.get('score', 0)
                
                if score >= min_score:
                    return question, answer
            
            # Format 3: Body and reply
            elif 'body' in data and 'reply' in data:
                question = data['body'].strip()
                answer = data['reply'].strip()
                score = data.get('score', 0)
                
                if score >= min_score:
                    return question, answer
            
            return None, None
            
        except Exception as e:
            logger.debug(f"Error extracting conversation: {e}")
            return None, None
    
    def load_sample(
        self,
        file_path: str,
        sample_size: int = 100
    ) -> List[Tuple[str, str]]:
        """
        Load a sample of conversations for testing.
        
        Args:
            file_path: Path to JSON file
            sample_size: Number of conversations to load
            
        Returns:
            List of (question, answer) tuples
        """
        return self.load_from_file(file_path, max_conversations=sample_size)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            'conversations_loaded': self.conversations_loaded
        }


# Example usage and test data generator
def generate_test_data(output_file: str, num_samples: int = 100):
    """
    Generate test Reddit-style data for development.
    
    Args:
        output_file: Path to output JSON file
        num_samples: Number of samples to generate
    """
    import random
    
    # Sample conversations
    samples = [
        {"question": "What's your favorite programming language?", "answer": "I really like Python for its simplicity and readability.", "score": 5},
        {"question": "How do I learn machine learning?", "answer": "Start with the basics of Python, then move to libraries like scikit-learn and TensorFlow.", "score": 10},
        {"question": "What's the best way to debug code?", "answer": "Use a debugger, add print statements, and read error messages carefully.", "score": 7},
        {"question": "Why is my code slow?", "answer": "Could be many reasons - inefficient algorithms, unnecessary loops, or memory issues. Profile your code to find bottlenecks.", "score": 8},
        {"question": "What is recursion?", "answer": "Recursion is when a function calls itself to solve a problem by breaking it into smaller subproblems.", "score": 12},
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            sample = random.choice(samples)
            # Add some variation
            sample_copy = sample.copy()
            sample_copy['score'] = random.randint(1, 15)
            f.write(json.dumps(sample_copy) + '\n')
    
    logger.info(f"Generated {num_samples} test samples in {output_file}")
