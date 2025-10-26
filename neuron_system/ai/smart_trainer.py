"""
Smart Training System - Intelligent data ingestion with deduplication and quality control.

This system ensures:
- No duplicate neurons
- Quality filtering
- Logic validation
- Efficient scaling
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SmartTrainer:
    """
    Intelligent training system with deduplication and quality control.
    """
    
    def __init__(self, language_model):
        """
        Initialize smart trainer.
        
        Args:
            language_model: LanguageModel instance
        """
        self.language_model = language_model
        self.graph = language_model.graph
        self.compression_engine = language_model.compression_engine
        
        # Quality thresholds
        self.min_text_length = 10  # Minimum characters
        self.max_text_length = 1000  # Maximum characters
        self.similarity_threshold = 0.95  # 95% similar = duplicate
        self.min_quality_score = 0.5  # Minimum quality score
        
        # Statistics
        self.total_processed = 0
        self.duplicates_found = 0
        self.quality_rejected = 0
        self.logic_rejected = 0
        self.successfully_added = 0
        
        # Bad patterns to reject
        self.bad_patterns = [
            '[deleted]',
            '[removed]',
            'http://',
            'https://',
            '&gt;',
            '&lt;',
            '&amp;',
        ]
    
    def train_conversation(
        self,
        question: str,
        answer: str,
        context: Optional[Dict[str, Any]] = None,
        check_duplicates: bool = True,
        check_quality: bool = True
    ) -> Tuple[bool, str]:
        """
        Train on a conversation pair with intelligent checks.
        
        Args:
            question: Question text
            answer: Answer text
            context: Optional context metadata
            check_duplicates: Whether to check for duplicates
            check_quality: Whether to check quality
            
        Returns:
            Tuple of (success, reason)
        """
        self.total_processed += 1
        
        # Step 1: Quality check
        if check_quality:
            is_quality, reason = self._check_quality(question, answer)
            if not is_quality:
                self.quality_rejected += 1
                logger.debug(f"Quality rejected: {reason}")
                return False, f"Quality: {reason}"
        
        # Step 2: Check for duplicates
        if check_duplicates:
            is_duplicate, existing_id = self._check_duplicate(question, answer)
            if is_duplicate:
                self.duplicates_found += 1
                logger.debug(f"Duplicate found: {existing_id}")
                return False, f"Duplicate of {existing_id}"
        
        # Step 3: Logic validation
        is_valid, reason = self._validate_logic(question, answer)
        if not is_valid:
            self.logic_rejected += 1
            logger.debug(f"Logic rejected: {reason}")
            return False, f"Logic: {reason}"
        
        # Step 4: Create neuron
        try:
            text = f"Question: {question} Answer: {answer}"
            tags = ['conversation', 'qa']
            if context:
                tags.extend(context.get('tags', []))
            
            neuron_id = self.language_model.learn(
                text=text,
                tags=tags,
                create_connections=True
            )
            
            self.successfully_added += 1
            logger.debug(f"Added neuron: {neuron_id}")
            return True, f"Added: {neuron_id}"
            
        except Exception as e:
            logger.error(f"Failed to create neuron: {e}")
            return False, f"Error: {e}"
    
    def _check_quality(self, question: str, answer: str) -> Tuple[bool, str]:
        """
        Check if conversation meets quality standards.
        
        Args:
            question: Question text
            answer: Answer text
            
        Returns:
            Tuple of (is_quality, reason)
        """
        # Check length
        if len(question) < self.min_text_length:
            return False, "Question too short"
        
        if len(answer) < self.min_text_length:
            return False, "Answer too short"
        
        if len(question) > self.max_text_length:
            return False, "Question too long"
        
        if len(answer) > self.max_text_length:
            return False, "Answer too long"
        
        # Check for bad patterns
        for pattern in self.bad_patterns:
            if pattern in question.lower() or pattern in answer.lower():
                return False, f"Contains bad pattern: {pattern}"
        
        # Check if answer is just a question
        if answer.strip().endswith('?') and len(answer) < 50:
            return False, "Answer is just a question"
        
        # Check if answer is too generic
        generic_answers = [
            'yes', 'no', 'ok', 'lol', 'haha', 'idk', 'dunno',
            'maybe', 'sure', 'nope', 'yep', 'yeah', 'nah'
        ]
        if answer.strip().lower() in generic_answers:
            return False, "Answer too generic"
        
        # Check quality score
        quality_score = self._calculate_quality_score(question, answer)
        if quality_score < self.min_quality_score:
            return False, f"Quality score too low: {quality_score:.2f}"
        
        return True, "Quality OK"
    
    def _calculate_quality_score(self, question: str, answer: str) -> float:
        """
        Calculate quality score for conversation.
        
        Args:
            question: Question text
            answer: Answer text
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        score = 0.5  # Start at 0.5
        
        # Bonus for good length
        if 20 <= len(answer) <= 200:
            score += 0.1
        
        # Bonus for proper capitalization
        if answer[0].isupper():
            score += 0.05
        
        # Bonus for punctuation
        if answer.strip()[-1] in '.!?':
            score += 0.05
        
        # Bonus for multiple sentences
        if answer.count('.') > 1 or answer.count('!') > 0:
            score += 0.1
        
        # Bonus for relevance (word overlap)
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        common_words = {'the', 'a', 'an', 'is', 'are', 'to', 'of', 'in', 'on'}
        q_words -= common_words
        a_words -= common_words
        
        if q_words:
            overlap = len(q_words & a_words) / len(q_words)
            score += overlap * 0.2
        
        return min(1.0, score)
    
    def _check_duplicate(self, question: str, answer: str) -> Tuple[bool, Optional[UUID]]:
        """
        Check if similar conversation already exists.
        
        Args:
            question: Question text
            answer: Answer text
            
        Returns:
            Tuple of (is_duplicate, existing_neuron_id)
        """
        try:
            # Compress the new conversation
            text = f"Question: {question} Answer: {answer}"
            new_vector, _ = self.compression_engine.compress(text)
            
            # Search for similar neurons
            results = self.language_model.understand(
                text,
                top_k=5,
                propagation_depth=0
            )
            
            if not results:
                return False, None
            
            # Check similarity of top result
            top_result = results[0]
            if top_result.activation >= self.similarity_threshold:
                # Very similar - likely duplicate
                return True, top_result.neuron.id
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return False, None
    
    def _validate_logic(self, question: str, answer: str) -> Tuple[bool, str]:
        """
        Validate logical consistency of conversation.
        
        Args:
            question: Question text
            answer: Answer text
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check for contradictions
        if 'yes' in answer.lower() and 'no' in answer.lower():
            # Both yes and no in answer - might be contradictory
            if len(answer) < 50:  # Short answer with both
                return False, "Contradictory answer (yes and no)"
        
        # Check if answer addresses the question
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # If question asks "what", answer should not be just "yes/no"
        if 'what' in question_lower:
            if answer_lower.strip() in ['yes', 'no', 'yes.', 'no.']:
                return False, "What-question answered with yes/no"
        
        # If question asks "who", answer should mention a person/entity
        if 'who' in question_lower:
            if len(answer) < 15:
                return False, "Who-question with too short answer"
        
        # Check for nonsensical patterns
        nonsense_patterns = [
            'asdf', 'qwerty', '!!!!!!', '??????',
            'aaaa', 'hhhh', 'mmmm'
        ]
        for pattern in nonsense_patterns:
            if pattern in answer_lower:
                return False, f"Nonsensical pattern: {pattern}"
        
        return True, "Logic OK"
    
    def batch_train(
        self,
        conversations: List[Tuple[str, str]],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Train on multiple conversations with batching.
        
        Args:
            conversations: List of (question, answer) tuples
            batch_size: Number to process before saving
            show_progress: Whether to show progress
            
        Returns:
            Statistics dictionary
        """
        total = len(conversations)
        processed = 0
        
        for i, (question, answer) in enumerate(conversations):
            success, reason = self.train_conversation(question, answer)
            
            processed += 1
            
            # Show progress
            if show_progress and processed % batch_size == 0:
                logger.info(
                    f"Processed {processed}/{total} "
                    f"(Added: {self.successfully_added}, "
                    f"Duplicates: {self.duplicates_found}, "
                    f"Rejected: {self.quality_rejected + self.logic_rejected})"
                )
        
        return self.get_statistics()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'total_processed': self.total_processed,
            'successfully_added': self.successfully_added,
            'duplicates_found': self.duplicates_found,
            'quality_rejected': self.quality_rejected,
            'logic_rejected': self.logic_rejected,
            'total_rejected': self.duplicates_found + self.quality_rejected + self.logic_rejected,
            'success_rate': self.successfully_added / self.total_processed if self.total_processed > 0 else 0.0
        }
    
    def reset_statistics(self):
        """Reset training statistics."""
        self.total_processed = 0
        self.duplicates_found = 0
        self.quality_rejected = 0
        self.logic_rejected = 0
        self.successfully_added = 0
