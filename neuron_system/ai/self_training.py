"""
Self-Training System - AI learns from its own neurons and improves continuously.

This system enables the AI to:
- Learn from successful responses
- Reinforce neurons that produce good answers
- Create new connections based on success patterns
- Weaken neurons that produce poor answers
- Continuously improve without manual training
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class SelfTraining:
    """
    Self-training system for continuous improvement.
    """
    
    def __init__(self, language_model):
        """
        Initialize self-training system.
        
        Args:
            language_model: LanguageModel instance
        """
        self.language_model = language_model
        self.graph = language_model.graph
        self.compression_engine = language_model.compression_engine
        
        # Track neuron performance
        self.neuron_success_count = {}  # neuron_id -> success_count
        self.neuron_failure_count = {}  # neuron_id -> failure_count
        self.neuron_last_used = {}  # neuron_id -> timestamp
        
        # Track response diversity
        self.response_to_queries = {}  # response_hash -> list of queries
        self.query_to_response = {}  # query -> response
        self.generic_responses = set()  # Set of generic/overused responses
        
        # Learning parameters - AGGRESSIVE SETTINGS
        self.success_boost = 0.2  # Stronger boost for good neurons
        self.failure_penalty = 0.3  # MUCH stronger penalty for bad neurons
        self.min_importance = 0.0  # Allow neurons to reach zero (for removal)
        self.max_importance = 1.0  # Maximum neuron importance
        self.max_response_reuse = 2  # Stricter - only 2 reuses allowed
        self.removal_threshold = 0.15  # Remove neurons below this importance
        self.synapse_weakening = 0.2  # How much to weaken bad synapses
        
        # Statistics
        self.total_feedback = 0
        self.positive_feedback = 0
        self.negative_feedback = 0
        self.neurons_reinforced = 0
        self.neurons_weakened = 0
        self.neurons_removed = 0  # Track removed neurons
        self.synapses_removed = 0  # Track removed synapses
        self.new_connections_created = 0
        self.generic_responses_detected = 0
        self.diversity_violations = 0
    
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
            feedback: "positive", "negative", or "auto" (auto-evaluate)
        """
        self.total_feedback += 1
        
        # Check response diversity
        diversity_ok = self._check_response_diversity(query, response)
        if not diversity_ok:
            feedback = "negative"  # Force negative feedback for diversity violations
            self.diversity_violations += 1
            logger.warning(f"Response diversity violation: same response for different queries")
        
        # Auto-evaluate if needed
        if feedback == "auto":
            feedback = self._auto_evaluate(query, response)
        
        # Track query-response pairs
        self._track_response(query, response)
        
        # Apply learning based on feedback
        if feedback == "positive":
            self._reinforce_neurons(activated_neurons, query, response)
            self.positive_feedback += 1
        elif feedback == "negative":
            self._weaken_neurons(activated_neurons)
            self.negative_feedback += 1
        
        # Update usage timestamps
        for neuron in activated_neurons:
            self.neuron_last_used[neuron.neuron.id] = datetime.now()
    
    def _check_response_diversity(self, query: str, response: str) -> bool:
        """
        Check if response is diverse enough (not reused too much).
        
        Args:
            query: Current query
            response: Current response
            
        Returns:
            True if diverse, False if overused
        """
        # Create hash of response (normalized)
        response_hash = response.strip().lower()[:100]  # First 100 chars
        
        # Check if this response has been used before
        if response_hash in self.response_to_queries:
            queries = self.response_to_queries[response_hash]
            
            # Check if used for different queries
            different_queries = [q for q in queries if q.lower() != query.lower()]
            
            if len(different_queries) >= self.max_response_reuse:
                # This response is overused
                self.generic_responses.add(response_hash)
                logger.warning(
                    f"Generic response detected (used {len(different_queries)} times): "
                    f"{response[:50]}..."
                )
                return False
        
        return True
    
    def _track_response(self, query: str, response: str):
        """
        Track query-response pairs for diversity monitoring.
        
        Args:
            query: Query
            response: Response
        """
        response_hash = response.strip().lower()[:100]
        
        # Track response to queries mapping
        if response_hash not in self.response_to_queries:
            self.response_to_queries[response_hash] = []
        self.response_to_queries[response_hash].append(query)
        
        # Track query to response mapping
        self.query_to_response[query] = response
    
    def _auto_evaluate(self, query: str, response: str) -> str:
        """
        Automatically evaluate response quality - INTELLIGENT VERSION.
        
        Args:
            query: Original query
            response: Generated response
            
        Returns:
            "positive" or "negative"
        """
        response_lower = response.lower()
        query_lower = query.lower()
        
        # CRITICAL FAILURES (immediate negative)
        critical_bad_patterns = [
            "I don't have enough knowledge",
            "I'm not confident in my answer",
            "Could you rephrase the question",
            "I don't have relevant information"
        ]
        
        if any(pattern.lower() in response_lower for pattern in critical_bad_patterns):
            logger.debug(f"Critical failure pattern detected")
            return "negative"
        
        # Check for meta-instructions (training artifacts)
        meta_patterns = [
            "when asked", "when someone", "means asking", 
            "means requesting", "respond with", "= = "
        ]
        if any(pattern in response_lower for pattern in meta_patterns):
            logger.debug(f"Meta-instruction pattern detected")
            return "negative"
        
        # Check if response is generic/overused
        response_hash = response.strip().lower()[:100]
        if response_hash in self.generic_responses:
            logger.debug("Generic response detected")
            return "negative"
        
        # MINIMUM LENGTH CHECK (but be reasonable)
        if len(response.strip()) < 15:
            logger.debug(f"Response too short: {len(response.strip())} chars")
            return "negative"
        
        # POSITIVE INDICATORS (signs of good response)
        positive_score = 0
        
        # 1. Good length (substantial answer)
        if len(response) >= 30:
            positive_score += 1
        if len(response) >= 50:
            positive_score += 1
        
        # 2. Contains specific information (not just generic phrases)
        specific_indicators = [
            'is', 'are', 'can', 'will', 'help', 'provide', 'assist',
            'system', 'designed', 'able', 'answer', 'questions'
        ]
        if sum(1 for word in specific_indicators if word in response_lower) >= 3:
            positive_score += 1
        
        # 3. Directly addresses the query
        query_words = set(query_lower.split())
        response_words = set(response_lower.split())
        common_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'can', 'could', 'should', 'may', 'might', 'must', 'to', 'of',
            'in', 'on', 'at', 'for', 'with', 'from', 'by', 'about', 'i',
            'you', 'me', 'my', 'your', 'that', 'this', 'it', 'what', 'who'
        }
        query_words -= common_words
        response_words -= common_words
        
        if query_words:
            overlap = len(query_words & response_words)
            relevance = overlap / len(query_words)
            
            # Be more lenient with relevance
            if relevance >= 0.2:
                positive_score += 1
            if relevance >= 0.4:
                positive_score += 1
        
        # 4. Answers identity questions appropriately
        identity_questions = ['what are you', 'who are you', 'what is your name']
        if any(q in query_lower for q in identity_questions):
            identity_answers = ['ai', 'assistant', 'system', 'help', 'designed']
            if any(word in response_lower for word in identity_answers):
                positive_score += 2  # Strong positive for identity answers
        
        # 5. Provides explanations for "what is" questions
        if 'what is' in query_lower or 'what are' in query_lower:
            # Check if response provides a definition/explanation
            explanation_words = ['is', 'are', 'means', 'refers', 'describes', 'involves']
            if any(word in response_lower for word in explanation_words):
                positive_score += 1
        
        # 6. Not just a generic filler
        generic_only_phrases = [
            "i can help with that",
            "let me help you",
            "how can i assist",
            "what would you like to know"
        ]
        is_generic_only = any(phrase in response_lower for phrase in generic_only_phrases)
        if is_generic_only and len(response) < 40:
            positive_score -= 2  # Penalty for generic-only short responses
        
        # DECISION: Positive if score >= 2
        if positive_score >= 2:
            logger.debug(f"Response passed evaluation (score: {positive_score})")
            return "positive"
        else:
            logger.debug(f"Response failed evaluation (score: {positive_score})")
            return "negative"
    
    def _reinforce_neurons(
        self,
        activated_neurons: List[Any],
        query: str,
        response: str
    ):
        """
        Reinforce neurons that produced a good response.
        ENHANCED: Stronger reinforcement for consistently good neurons.
        
        Args:
            activated_neurons: Neurons to reinforce
            query: Original query
            response: Good response
        """
        for activated in activated_neurons:
            neuron = activated.neuron
            neuron_id = neuron.id
            
            # Track success
            self.neuron_success_count[neuron_id] = \
                self.neuron_success_count.get(neuron_id, 0) + 1
            
            # Calculate success ratio for this neuron
            success_count = self.neuron_success_count.get(neuron_id, 0)
            failure_count = self.neuron_failure_count.get(neuron_id, 0)
            total_uses = success_count + failure_count
            success_ratio = success_count / total_uses if total_uses > 0 else 0.5
            
            # Boost importance/activation
            if hasattr(neuron, 'importance'):
                old_importance = neuron.importance
                
                # ENHANCED: Stronger boost for consistently successful neurons
                boost = self.success_boost
                if success_ratio > 0.7 and total_uses >= 3:
                    # Extra boost for consistently good neurons
                    boost *= 1.5
                    logger.debug(f"Extra boost for consistent neuron {neuron_id} (ratio: {success_ratio:.2f})")
                
                neuron.importance = min(
                    self.max_importance,
                    neuron.importance + boost
                )
                
                if neuron.importance > old_importance:
                    self.neurons_reinforced += 1
                    logger.debug(
                        f"Reinforced neuron {neuron_id}: {old_importance:.2f} → {neuron.importance:.2f} "
                        f"(success: {success_count}/{total_uses})"
                    )
            
            # Create new connections to other successful neurons
            self._create_success_connections(neuron, activated_neurons)
            
            # Strengthen existing connections to this neuron
            self._strengthen_incoming_synapses(neuron_id)
    
    def _weaken_neurons(self, activated_neurons: List[Any]):
        """
        Weaken neurons that produced a poor response.
        AGGRESSIVE: Also weakens/removes connected synapses.
        
        Args:
            activated_neurons: Neurons to weaken
        """
        for activated in activated_neurons:
            neuron = activated.neuron
            neuron_id = neuron.id
            
            # Track failure
            self.neuron_failure_count[neuron_id] = \
                self.neuron_failure_count.get(neuron_id, 0) + 1
            
            # Reduce importance/activation AGGRESSIVELY
            if hasattr(neuron, 'importance'):
                old_importance = neuron.importance
                neuron.importance = max(
                    self.min_importance,
                    neuron.importance - self.failure_penalty
                )
                if neuron.importance < old_importance:
                    self.neurons_weakened += 1
                    logger.debug(f"Weakened neuron {neuron_id}: {old_importance:.2f} → {neuron.importance:.2f}")
                
                # REMOVE neuron if too weak
                if neuron.importance <= self.removal_threshold:
                    logger.warning(f"Removing weak neuron {neuron_id} (importance: {neuron.importance:.2f})")
                    self._remove_neuron(neuron_id)
            
            # WEAKEN connected synapses
            self._weaken_synapses(neuron_id)
    
    def _strengthen_incoming_synapses(self, neuron_id: UUID):
        """
        Strengthen synapses that lead to a successful neuron.
        
        Args:
            neuron_id: ID of the successful neuron
        """
        strengthened = 0
        for synapse in self.graph.synapses.values():
            # Find synapses pointing TO this neuron
            if synapse.target_neuron_id == neuron_id:
                old_weight = synapse.weight
                # Strengthen the connection
                synapse.weight = min(1.0, synapse.weight + 0.05)
                if synapse.weight > old_weight:
                    strengthened += 1
                    logger.debug(f"Strengthened synapse to {neuron_id}: {old_weight:.2f} → {synapse.weight:.2f}")
        
        if strengthened > 0:
            logger.debug(f"Strengthened {strengthened} incoming synapses to neuron {neuron_id}")
    
    def _remove_neuron(self, neuron_id: UUID):
        """
        Remove a neuron and all its synapses from the graph.
        
        Args:
            neuron_id: ID of neuron to remove
        """
        try:
            # Remove all synapses connected to this neuron
            synapses_to_remove = []
            for synapse_id, synapse in self.graph.synapses.items():
                if (synapse.source_neuron_id == neuron_id or 
                    synapse.target_neuron_id == neuron_id):
                    synapses_to_remove.append(synapse_id)
            
            for synapse_id in synapses_to_remove:
                self.graph.remove_synapse(synapse_id)
                self.synapses_removed += 1
            
            # Remove the neuron itself
            self.graph.remove_neuron(neuron_id)
            self.neurons_removed += 1
            
            # Clean up tracking data
            self.neuron_success_count.pop(neuron_id, None)
            self.neuron_failure_count.pop(neuron_id, None)
            self.neuron_last_used.pop(neuron_id, None)
            
            logger.info(f"Removed neuron {neuron_id} and {len(synapses_to_remove)} synapses")
        except Exception as e:
            logger.error(f"Failed to remove neuron {neuron_id}: {e}")
    
    def _weaken_synapses(self, neuron_id: UUID):
        """
        Weaken or remove synapses connected to a bad neuron.
        
        Args:
            neuron_id: ID of the bad neuron
        """
        synapses_to_remove = []
        
        for synapse_id, synapse in list(self.graph.synapses.items()):
            # Check if synapse is connected to this neuron
            if (synapse.source_neuron_id == neuron_id or 
                synapse.target_neuron_id == neuron_id):
                
                # Weaken the synapse
                old_weight = synapse.weight
                synapse.weight = synapse.weight - self.synapse_weakening
                
                logger.debug(f"Weakened synapse {synapse_id}: {old_weight:.2f} → {synapse.weight:.2f}")
                
                # Remove if too weak
                if abs(synapse.weight) < 0.1:
                    synapses_to_remove.append(synapse_id)
        
        # Remove weak synapses
        for synapse_id in synapses_to_remove:
            try:
                self.graph.remove_synapse(synapse_id)
                self.synapses_removed += 1
                logger.debug(f"Removed weak synapse {synapse_id}")
            except Exception as e:
                logger.error(f"Failed to remove synapse {synapse_id}: {e}")
    
    def _create_success_connections(
        self,
        neuron: Any,
        activated_neurons: List[Any]
    ):
        """
        Create connections between successful neurons.
        
        Args:
            neuron: Source neuron
            activated_neurons: Other activated neurons
        """
        # Find other successful neurons
        successful_neurons = [
            a.neuron for a in activated_neurons
            if self.neuron_success_count.get(a.neuron.id, 0) > 
               self.neuron_failure_count.get(a.neuron.id, 0)
        ]
        
        # Create connections to top successful neurons
        for target_neuron in successful_neurons[:3]:
            if target_neuron.id == neuron.id:
                continue
            
            # Check if connection already exists
            existing = False
            for synapse in self.graph.synapses.values():
                if (synapse.source_neuron_id == neuron.id and
                    synapse.target_neuron_id == target_neuron.id):
                    # Strengthen existing connection
                    old_weight = synapse.weight
                    synapse.weight = min(1.0, synapse.weight + 0.1)
                    existing = True
                    logger.debug(f"Strengthened synapse: {old_weight:.2f} → {synapse.weight:.2f}")
                    break
            
            if not existing:
                # Create new connection
                from neuron_system.core.synapse import Synapse
                synapse = Synapse(
                    source_neuron_id=neuron.id,
                    target_neuron_id=target_neuron.id,
                    weight=0.5  # Start with moderate weight
                )
                self.graph.add_synapse(synapse)
                self.new_connections_created += 1
                logger.debug(f"Created new synapse: {neuron.id} → {target_neuron.id}")
    
    def consolidate_learning(self):
        """
        Consolidate learning by analyzing patterns and optimizing network.
        AGGRESSIVE: Actually removes bad neurons and synapses.
        """
        logger.info("=" * 70)
        logger.info("CONSOLIDATING LEARNING (AGGRESSIVE MODE)")
        logger.info("=" * 70)
        
        # Find consistently successful neurons
        successful_neurons = []
        for neuron_id, success_count in self.neuron_success_count.items():
            failure_count = self.neuron_failure_count.get(neuron_id, 0)
            if success_count > failure_count * 2:  # At least 2:1 success ratio
                successful_neurons.append(neuron_id)
        
        # Find consistently failing neurons
        failing_neurons = []
        for neuron_id, failure_count in self.neuron_failure_count.items():
            success_count = self.neuron_success_count.get(neuron_id, 0)
            if failure_count > success_count:  # More failures than successes
                failing_neurons.append((neuron_id, failure_count, success_count))
        
        logger.info(f"Found {len(successful_neurons)} consistently successful neurons")
        logger.info(f"Found {len(failing_neurons)} consistently failing neurons")
        
        # Boost successful neurons further
        boosted = 0
        for neuron_id in successful_neurons:
            if neuron_id in self.graph.neurons:
                neuron = self.graph.neurons[neuron_id]
                if hasattr(neuron, 'importance'):
                    old_importance = neuron.importance
                    neuron.importance = min(self.max_importance, neuron.importance + 0.2)
                    boosted += 1
                    logger.debug(f"Boosted successful neuron {neuron_id}: {old_importance:.2f} → {neuron.importance:.2f}")
        
        logger.info(f"Boosted {boosted} successful neurons")
        
        # AGGRESSIVELY REMOVE failing neurons
        removed_count = 0
        for neuron_id, failures, successes in failing_neurons:
            if neuron_id not in self.graph.neurons:
                continue
            
            neuron = self.graph.neurons[neuron_id]
            
            # Remove if:
            # 1. Has low importance
            # 2. Has more failures than successes
            # 3. Has been used at least 3 times (to have enough data)
            total_uses = failures + successes
            if (hasattr(neuron, 'importance') and 
                neuron.importance < 0.3 and 
                failures > successes and
                total_uses >= 3):
                
                logger.warning(
                    f"REMOVING bad neuron {neuron_id}: "
                    f"failures={failures}, successes={successes}, "
                    f"importance={neuron.importance:.2f}"
                )
                self._remove_neuron(neuron_id)
                removed_count += 1
        
        logger.info(f"Removed {removed_count} consistently bad neurons")
        
        # Clean up weak synapses
        weak_synapses = []
        for synapse_id, synapse in list(self.graph.synapses.items()):
            if abs(synapse.weight) < 0.15:  # Very weak connection
                weak_synapses.append(synapse_id)
        
        for synapse_id in weak_synapses:
            try:
                self.graph.remove_synapse(synapse_id)
                self.synapses_removed += 1
            except Exception as e:
                logger.error(f"Failed to remove weak synapse: {e}")
        
        logger.info(f"Removed {len(weak_synapses)} weak synapses")
        logger.info("=" * 70)
        logger.info("CONSOLIDATION COMPLETE")
        logger.info("=" * 70)
    
    def generate_synthetic_examples(self, num_examples: int = 10):
        """
        Generate synthetic training examples from successful patterns.
        
        Args:
            num_examples: Number of examples to generate
        """
        logger.info(f"Generating {num_examples} synthetic examples...")
        
        # Find most successful neurons
        successful_neurons = sorted(
            self.neuron_success_count.items(),
            key=lambda x: x[1] - self.neuron_failure_count.get(x[0], 0),
            reverse=True
        )[:num_examples]
        
        synthetic_count = 0
        for neuron_id, _ in successful_neurons:
            if neuron_id not in self.graph.neurons:
                continue
            
            neuron = self.graph.neurons[neuron_id]
            if not hasattr(neuron, 'source_data') or not neuron.source_data:
                continue
            
            # Create variations of successful patterns
            source_text = neuron.source_data
            
            # Simple variation: if it's a Q&A, create similar Q&A
            if "Question:" in source_text and "Answer:" in source_text:
                # Extract and learn from this pattern
                self.language_model.learn(
                    text=source_text,
                    tags=getattr(neuron, 'semantic_tags', []) + ['synthetic', 'self-generated'],
                    create_connections=True
                )
                synthetic_count += 1
        
        logger.info(f"Generated {synthetic_count} synthetic examples")
        return synthetic_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get self-training statistics."""
        success_rate = (
            self.positive_feedback / self.total_feedback
            if self.total_feedback > 0 else 0.0
        )
        
        # Calculate improvement metrics
        net_neuron_change = self.neurons_reinforced - self.neurons_weakened
        net_quality = self.positive_feedback - self.negative_feedback
        
        return {
            'total_feedback': self.total_feedback,
            'positive_feedback': self.positive_feedback,
            'negative_feedback': self.negative_feedback,
            'success_rate': success_rate,
            'neurons_reinforced': self.neurons_reinforced,
            'neurons_weakened': self.neurons_weakened,
            'neurons_removed': self.neurons_removed,
            'synapses_removed': self.synapses_removed,
            'new_connections_created': self.new_connections_created,
            'tracked_neurons': len(self.neuron_success_count),
            'generic_responses_detected': self.generic_responses_detected,
            'diversity_violations': self.diversity_violations,
            'net_neuron_change': net_neuron_change,
            'net_quality': net_quality,
        }
    
    def save_learning_state(self):
        """
        Save learning state to database.
        ENHANCED: Saves ALL neurons and synapses, plus performance tracking.
        """
        logger.info("=" * 70)
        logger.info("SAVING LEARNING STATE TO DATABASE")
        logger.info("=" * 70)
        
        saved_neurons = 0
        saved_synapses = 0
        
        # Save ALL neurons (not just tracked ones)
        # This ensures importance values are persisted
        if hasattr(self.graph, 'neuron_store'):
            try:
                all_neurons = list(self.graph.neurons.values())
                logger.info(f"Saving {len(all_neurons)} neurons...")
                
                # Batch update in chunks for better performance
                chunk_size = 1000
                for i in range(0, len(all_neurons), chunk_size):
                    chunk = all_neurons[i:i+chunk_size]
                    self.graph.neuron_store.batch_update(chunk)
                    saved_neurons += len(chunk)
                    if i % 1000 == 0 and i > 0:
                        logger.info(f"  Saved {saved_neurons}/{len(all_neurons)} neurons...")
                
                logger.info(f"✓ Saved {saved_neurons} neurons to database")
            except Exception as e:
                logger.error(f"Failed to save neurons: {e}")
        else:
            logger.warning("No neuron_store available - neurons not saved!")
        
        # Save ALL synapses
        if hasattr(self.graph, 'synapse_store'):
            try:
                all_synapses = list(self.graph.synapses.values())
                logger.info(f"Saving {len(all_synapses)} synapses...")
                
                # Batch update in chunks
                chunk_size = 1000
                for i in range(0, len(all_synapses), chunk_size):
                    chunk = all_synapses[i:i+chunk_size]
                    self.graph.synapse_store.batch_update(chunk)
                    saved_synapses += len(chunk)
                    if i % 1000 == 0 and i > 0:
                        logger.info(f"  Saved {saved_synapses}/{len(all_synapses)} synapses...")
                
                logger.info(f"✓ Saved {saved_synapses} synapses to database")
            except Exception as e:
                logger.error(f"Failed to save synapses: {e}")
        else:
            logger.warning("No synapse_store available - synapses not saved!")
        
        # Save performance tracking data (for future analysis)
        logger.info(f"Performance tracking:")
        logger.info(f"  Tracked neurons: {len(self.neuron_success_count)}")
        logger.info(f"  Success records: {sum(self.neuron_success_count.values())}")
        logger.info(f"  Failure records: {sum(self.neuron_failure_count.values())}")
        
        logger.info("=" * 70)
        logger.info("SAVE COMPLETE")
        logger.info("=" * 70)


class ContinuousLearning:
    """
    Continuous learning system that runs in background.
    """
    
    def __init__(self, language_model):
        """
        Initialize continuous learning.
        
        Args:
            language_model: LanguageModel instance
        """
        self.language_model = language_model
        self.self_training = SelfTraining(language_model)
        self.learning_enabled = True
        self.consolidation_interval = 100  # Consolidate every N interactions
        self.interaction_count = 0
    
    def process_interaction(
        self,
        query: str,
        response: str,
        activated_neurons: List[Any],
        user_feedback: Optional[str] = None
    ):
        """
        Process an interaction for continuous learning.
        
        Args:
            query: User query
            response: AI response
            activated_neurons: Activated neurons
            user_feedback: Optional explicit user feedback
        """
        if not self.learning_enabled:
            return
        
        self.interaction_count += 1
        
        # Learn from this interaction
        feedback = user_feedback if user_feedback else "auto"
        self.self_training.learn_from_response(
            query, response, activated_neurons, feedback
        )
        
        # Periodic consolidation
        if self.interaction_count % self.consolidation_interval == 0:
            self.self_training.consolidate_learning()
            self.self_training.save_learning_state()
            logger.info(f"Consolidated learning after {self.interaction_count} interactions")
    
    def enable_learning(self):
        """Enable continuous learning."""
        self.learning_enabled = True
        logger.info("Continuous learning enabled")
    
    def disable_learning(self):
        """Disable continuous learning."""
        self.learning_enabled = False
        logger.info("Continuous learning disabled")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get continuous learning statistics."""
        stats = self.self_training.get_statistics()
        stats['interaction_count'] = self.interaction_count
        stats['learning_enabled'] = self.learning_enabled
        return stats
