"""
Training Module - Konsolidierte Training-Funktionalität.

Dieses Modul vereint alle Training-Komponenten:
- SmartTrainer: Intelligentes Training mit Quality Control
- IncrementalTrainer: Inkrementelle Updates ohne Neutraining
- SelfTraining: Kontinuierliche Selbstverbesserung

Ersetzt die alten Files:
- smart_trainer.py
- incremental_trainer.py  
- self_training.py
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from uuid import UUID
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# SHARED UTILITIES
# ============================================================================

class QualityChecker:
    """
    Gemeinsame Quality-Checking Logik für alle Trainer.
    Vermeidet Code-Duplikation.
    """
    
    def __init__(self):
        self.min_text_length = 10
        self.max_text_length = 1000
        self.bad_patterns = [
            '[deleted]', '[removed]', 'http://', 'https://',
            '&gt;', '&lt;', '&amp;'
        ]
        self.generic_answers = [
            'yes', 'no', 'ok', 'lol', 'haha', 'idk', 'dunno',
            'maybe', 'sure', 'nope', 'yep', 'yeah', 'nah'
        ]
    
    def check_length(self, text: str) -> Tuple[bool, str]:
        """Check if text length is valid."""
        if len(text) < self.min_text_length:
            return False, "Text too short"
        if len(text) > self.max_text_length:
            return False, "Text too long"
        return True, "OK"
    
    def check_patterns(self, text: str) -> Tuple[bool, str]:
        """Check for bad patterns."""
        text_lower = text.lower()
        for pattern in self.bad_patterns:
            if pattern in text_lower:
                return False, f"Contains bad pattern: {pattern}"
        return True, "OK"
    
    def check_generic(self, text: str) -> Tuple[bool, str]:
        """Check if text is too generic."""
        if text.strip().lower() in self.generic_answers:
            return False, "Too generic"
        return True, "OK"
    
    def calculate_quality_score(self, question: str, answer: str) -> float:
        """Calculate overall quality score (0.0 to 1.0)."""
        score = 0.5
        
        # Length bonus
        if 20 <= len(answer) <= 200:
            score += 0.1
        
        # Capitalization
        if answer and answer[0].isupper():
            score += 0.05
        
        # Punctuation
        if answer and answer.strip()[-1] in '.!?':
            score += 0.05
        
        # Multiple sentences
        if answer.count('.') > 1 or answer.count('!') > 0:
            score += 0.1
        
        # Word overlap (relevance)
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        common = {'the', 'a', 'an', 'is', 'are', 'to', 'of', 'in', 'on'}
        q_words -= common
        a_words -= common
        
        if q_words:
            overlap = len(q_words & a_words) / len(q_words)
            score += overlap * 0.2
        
        return min(1.0, score)


class SimilarityFinder:
    """
    Gemeinsame Similarity-Search Logik.
    Vermeidet Code-Duplikation zwischen Trainern.
    """
    
    @staticmethod
    def find_similar(
        language_model,
        text: str,
        threshold: float = 0.95,
        top_k: int = 5
    ) -> Optional[Any]:
        """
        Find similar existing knowledge.
        
        Args:
            language_model: LanguageModel instance
            text: Text to search for
            threshold: Similarity threshold
            top_k: Number of results to check
            
        Returns:
            Similar neuron if found, None otherwise
        """
        try:
            results = language_model.understand(
                text, top_k=top_k, propagation_depth=0
            )
            
            if results and results[0].activation >= threshold:
                return results[0].neuron
            
            return None
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return None


# ============================================================================
# SMART TRAINER
# ============================================================================

class SmartTrainer:
    """
    Intelligentes Training mit Quality Control und Duplikat-Erkennung.
    
    Features:
    - Quality filtering
    - Duplicate detection
    - Logic validation
    - Batch processing
    """
    
    def __init__(self, language_model):
        """
        Initialize SmartTrainer.
        
        Args:
            language_model: LanguageModel instance
        """
        self.language_model = language_model
        self.graph = language_model.graph
        self.compression_engine = language_model.compression_engine
        
        # Shared utilities
        self.quality_checker = QualityChecker()
        self.similarity_finder = SimilarityFinder()
        
        # Thresholds
        self.similarity_threshold = 0.95
        self.min_quality_score = 0.5
        
        # Statistics
        self.total_processed = 0
        self.successfully_added = 0
        self.duplicates_found = 0
        self.quality_rejected = 0
        self.logic_rejected = 0
    
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
        
        # Quality check
        if check_quality:
            is_quality, reason = self._check_quality(question, answer)
            if not is_quality:
                self.quality_rejected += 1
                return False, f"Quality: {reason}"
        
        # Duplicate check
        if check_duplicates:
            text = f"Question: {question} Answer: {answer}"
            similar = self.similarity_finder.find_similar(
                self.language_model, text, self.similarity_threshold
            )
            if similar:
                self.duplicates_found += 1
                return False, f"Duplicate of {similar.id}"
        
        # Logic validation
        is_valid, reason = self._validate_logic(question, answer)
        if not is_valid:
            self.logic_rejected += 1
            return False, f"Logic: {reason}"
        
        # Create neuron
        try:
            text = f"Question: {question} Answer: {answer}"
            tags = ['conversation', 'qa']
            if context:
                tags.extend(context.get('tags', []))
            
            neuron_id = self.language_model.learn(
                text=text, tags=tags, create_connections=True
            )
            
            self.successfully_added += 1
            return True, f"Added: {neuron_id}"
            
        except Exception as e:
            logger.error(f"Failed to create neuron: {e}")
            return False, f"Error: {e}"
    
    def _check_quality(self, question: str, answer: str) -> Tuple[bool, str]:
        """Check conversation quality."""
        # Length check
        ok, reason = self.quality_checker.check_length(question)
        if not ok:
            return False, f"Question {reason}"
        
        ok, reason = self.quality_checker.check_length(answer)
        if not ok:
            return False, f"Answer {reason}"
        
        # Pattern check
        ok, reason = self.quality_checker.check_patterns(question)
        if not ok:
            return False, reason
        
        ok, reason = self.quality_checker.check_patterns(answer)
        if not ok:
            return False, reason
        
        # Generic check
        ok, reason = self.quality_checker.check_generic(answer)
        if not ok:
            return False, reason
        
        # Quality score
        score = self.quality_checker.calculate_quality_score(question, answer)
        if score < self.min_quality_score:
            return False, f"Quality score too low: {score:.2f}"
        
        return True, "Quality OK"
    
    def _validate_logic(self, question: str, answer: str) -> Tuple[bool, str]:
        """Validate logical consistency."""
        answer_lower = answer.lower()
        question_lower = question.lower()
        
        # Contradiction check
        if 'yes' in answer_lower and 'no' in answer_lower and len(answer) < 50:
            return False, "Contradictory answer"
        
        # Question type validation
        if 'what' in question_lower:
            if answer_lower.strip() in ['yes', 'no', 'yes.', 'no.']:
                return False, "What-question answered with yes/no"
        
        if 'who' in question_lower and len(answer) < 15:
            return False, "Who-question with too short answer"
        
        # Nonsense patterns
        nonsense = ['asdf', 'qwerty', '!!!!!!', '??????']
        for pattern in nonsense:
            if pattern in answer_lower:
                return False, f"Nonsensical pattern: {pattern}"
        
        return True, "Logic OK"
    
    def batch_train(
        self,
        conversations: List[Tuple[str, str]],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Batch train multiple conversations."""
        total = len(conversations)
        
        for i, (question, answer) in enumerate(conversations):
            self.train_conversation(question, answer)
            
            if show_progress and (i + 1) % batch_size == 0:
                logger.info(
                    f"Processed {i + 1}/{total} "
                    f"(Added: {self.successfully_added}, "
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
            'success_rate': (
                self.successfully_added / self.total_processed
                if self.total_processed > 0 else 0.0
            )
        }
    
    def reset_statistics(self):
        """Reset training statistics."""
        self.total_processed = 0
        self.successfully_added = 0
        self.duplicates_found = 0
        self.quality_rejected = 0
        self.logic_rejected = 0


# ============================================================================
# INCREMENTAL TRAINER
# ============================================================================

class IncrementalTrainer:
    """
    Inkrementelles Training ohne komplettes Neutraining.
    
    Features:
    - Add or update knowledge
    - Similarity-based deduplication
    - Connection updates
    - Batch operations
    """
    
    def __init__(self, language_model):
        """
        Initialize IncrementalTrainer.
        
        Args:
            language_model: LanguageModel instance
        """
        self.language_model = language_model
        self.existing_texts: Set[str] = set()
        self.text_to_neuron: Dict[str, UUID] = {}
        
        # Shared utilities
        self.similarity_finder = SimilarityFinder()
        
        # Build index
        self._build_knowledge_index()
    
    def _build_knowledge_index(self):
        """Build index of existing knowledge."""
        logger.info("Building knowledge index...")
        
        for neuron in self.language_model.graph.neurons.values():
            if hasattr(neuron, 'source_data') and neuron.source_data:
                text = neuron.source_data.strip()
                self.existing_texts.add(text)
                self.text_to_neuron[text] = neuron.id
        
        logger.info(f"Indexed {len(self.existing_texts)} items")
    
    def add_or_update_knowledge(
        self,
        text: str,
        tags: Optional[List[str]] = None,
        force_update: bool = False
    ) -> Tuple[UUID, bool]:
        """
        Add new knowledge or update existing.
        
        Args:
            text: Knowledge text
            tags: Semantic tags
            force_update: Force update even if exists
            
        Returns:
            Tuple of (neuron_id, was_updated)
        """
        text = text.strip()
        
        # Check exact match
        if text in self.existing_texts and not force_update:
            return self.text_to_neuron[text], False
        
        # Check similarity
        similar = self.similarity_finder.find_similar(
            self.language_model, text, threshold=0.95
        )
        
        if similar and not force_update:
            # Update tags if provided
            if tags and hasattr(similar, 'semantic_tags'):
                existing_tags = similar.semantic_tags or []
                similar.semantic_tags = list(set(existing_tags + tags))
            return similar.id, True
        
        # Add new
        neuron_id = self.language_model.learn(
            text=text, tags=tags, create_connections=True
        )
        
        # Update index
        self.existing_texts.add(text)
        self.text_to_neuron[text] = neuron_id
        
        return neuron_id, False
    
    def batch_add_or_update(
        self,
        knowledge_items: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, int]:
        """Batch add or update multiple items."""
        stats = {'added': 0, 'updated': 0, 'skipped': 0}
        
        for i, item in enumerate(knowledge_items):
            text = item.get('text', '')
            tags = item.get('tags', [])
            
            if not text or len(text) < 10:
                stats['skipped'] += 1
                continue
            
            neuron_id, was_updated = self.add_or_update_knowledge(text, tags)
            
            if was_updated:
                stats['updated'] += 1
            else:
                stats['added'] += 1
            
            if (i + 1) % batch_size == 0:
                logger.info(f"Processed {i + 1}/{len(knowledge_items)}")
        
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            'total_knowledge_items': len(self.existing_texts),
            'total_neurons': len(self.language_model.graph.neurons),
            'total_synapses': len(self.language_model.graph.synapses),
        }


# ============================================================================
# SELF TRAINING (Simplified - Core functionality only)
# ============================================================================

class SelfTraining:
    """
    Kontinuierliche Selbstverbesserung.
    
    Vereinfachte Version - fokussiert auf Core-Funktionalität.
    Für vollständige Version siehe self_training.py (deprecated).
    """
    
    def __init__(self, language_model):
        """Initialize SelfTraining."""
        self.language_model = language_model
        self.graph = language_model.graph
        
        # Track performance
        self.neuron_success_count = {}
        self.neuron_failure_count = {}
        
        # Parameters
        self.success_boost = 0.2
        self.failure_penalty = 0.3
        
        # Statistics
        self.total_feedback = 0
        self.positive_feedback = 0
        self.negative_feedback = 0
    
    def learn_from_response(
        self,
        query: str,
        response: str,
        activated_neurons: List[Any],
        feedback: str = "auto"
    ):
        """
        Learn from a response by reinforcing or weakening neurons.
        
        Args:
            query: Original query
            response: Generated response
            activated_neurons: Neurons that were activated
            feedback: "positive", "negative", or "auto"
        """
        self.total_feedback += 1
        
        # Auto-evaluate if needed
        if feedback == "auto":
            feedback = self._auto_evaluate(query, response)
        
        # Apply learning
        if feedback == "positive":
            self._reinforce_neurons(activated_neurons)
            self.positive_feedback += 1
        elif feedback == "negative":
            self._weaken_neurons(activated_neurons)
            self.negative_feedback += 1
    
    def _auto_evaluate(self, query: str, response: str) -> str:
        """Auto-evaluate response quality."""
        response_lower = response.lower()
        
        # Critical failures
        bad_patterns = [
            "I don't have enough knowledge",
            "I'm not confident",
            "Could you rephrase"
        ]
        
        if any(p.lower() in response_lower for p in bad_patterns):
            return "negative"
        
        # Positive indicators
        if len(response) >= 30 and any(c.isalpha() for c in response):
            return "positive"
        
        return "negative"
    
    def _reinforce_neurons(self, activated_neurons: List[Any]):
        """Reinforce successful neurons."""
        for activated in activated_neurons:
            neuron = activated.neuron
            neuron_id = neuron.id
            
            self.neuron_success_count[neuron_id] = \
                self.neuron_success_count.get(neuron_id, 0) + 1
            
            if hasattr(neuron, 'importance'):
                neuron.importance = min(1.0, neuron.importance + self.success_boost)
    
    def _weaken_neurons(self, activated_neurons: List[Any]):
        """Weaken unsuccessful neurons."""
        for activated in activated_neurons:
            neuron = activated.neuron
            neuron_id = neuron.id
            
            self.neuron_failure_count[neuron_id] = \
                self.neuron_failure_count.get(neuron_id, 0) + 1
            
            if hasattr(neuron, 'importance'):
                neuron.importance = max(0.0, neuron.importance - self.failure_penalty)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get self-training statistics."""
        return {
            'total_feedback': self.total_feedback,
            'positive_feedback': self.positive_feedback,
            'negative_feedback': self.negative_feedback,
            'success_rate': (
                self.positive_feedback / self.total_feedback
                if self.total_feedback > 0 else 0.0
            )
        }


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# Export all classes for backward compatibility
__all__ = [
    'SmartTrainer',
    'IncrementalTrainer',
    'SelfTraining',
    'QualityChecker',
    'SimilarityFinder',
]
