"""
Language Model built on the 3D Synaptic Neuron System.

Provides AI capabilities for natural language understanding and generation
using pre-trained knowledge neurons.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
import numpy as np

from neuron_system.core.graph import NeuronGraph
from neuron_system.core.vector3d import Vector3D
from neuron_system.core.synapse import Synapse
from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
from neuron_system.engines.compression import CompressionEngine
from neuron_system.engines.query import QueryEngine, ActivatedNeuron
from neuron_system.engines.training import TrainingEngine

logger = logging.getLogger(__name__)


class LanguageModel:
    """
    AI Language Model built on the neuron system.
    
    Provides natural language understanding and generation capabilities
    using a network of knowledge neurons.
    """
    
    def __init__(
        self,
        graph: NeuronGraph,
        compression_engine: CompressionEngine,
        query_engine: QueryEngine,
        training_engine: TrainingEngine,
        enable_self_training: bool = True
    ):
        """
        Initialize the language model.
        
        Args:
            graph: Neuron graph instance
            compression_engine: Compression engine for text encoding
            query_engine: Query engine for retrieval
            training_engine: Training engine for learning
            enable_self_training: Enable continuous self-training
        """
        self.graph = graph
        self.compression_engine = compression_engine
        self.query_engine = query_engine
        self.training_engine = training_engine
        
        # Initialize self-training
        if enable_self_training:
            from neuron_system.ai.self_training import ContinuousLearning
            self._continuous_learning = ContinuousLearning(self)
            logger.info("Language model initialized with self-training enabled")
        else:
            self._continuous_learning = None
            logger.info("Language model initialized")
    
    def understand(
        self,
        text: str,
        top_k: int = 10,
        propagation_depth: int = 3
    ) -> List[ActivatedNeuron]:
        """
        Understand input text by finding relevant knowledge neurons.
        
        Args:
            text: Input text to understand
            top_k: Number of top results to return
            propagation_depth: Depth of activation propagation
            
        Returns:
            List of query results with activated neurons
        """
        logger.info(f"Understanding text: '{text[:50]}...'")
        
        results = self.query_engine.query(
            query_text=text,
            top_k=top_k,
            propagation_depth=propagation_depth
        )
        
        logger.info(f"Found {len(results)} relevant neurons")
        return results
    
    def generate_response(
        self,
        query: str,
        context_size: int = 5,
        min_activation: float = 0.3,
        propagation_depth: int = 0,
        use_reasoning: bool = True
    ) -> str:
        """
        Generate a response to a query using activated neurons.
        
        Args:
            query: Input query
            context_size: Number of neurons to use for context
            min_activation: Minimum activation threshold
            propagation_depth: Depth of activation propagation (0 = no propagation)
            use_reasoning: Whether to use reasoning neurons to improve response
            
        Returns:
            Generated response text
        """
        logger.info(f"Generating response for: '{query[:50]}...'")
        
        # Find relevant neurons (without propagation for more accurate results)
        results = self.understand(query, top_k=context_size, propagation_depth=propagation_depth)
        
        if not results:
            return "I don't have enough knowledge to answer that question."
        
        # Filter by activation threshold
        relevant_results = [r for r in results if r.activation >= min_activation]
        
        if not relevant_results:
            return "I'm not confident enough to answer that question."
        
        # Extract knowledge from top neurons
        knowledge_pieces = []
        for result in relevant_results[:context_size]:
            neuron = result.neuron
            if hasattr(neuron, 'source_data') and neuron.source_data:
                knowledge_pieces.append({
                    'text': neuron.source_data,
                    'activation': result.activation,
                    'tags': getattr(neuron, 'semantic_tags', [])
                })
        
        # Apply reasoning if enabled
        if use_reasoning:
            knowledge_pieces = self._apply_reasoning(query, knowledge_pieces)
        
        # Generate response based on activated knowledge
        response = self._synthesize_response(query, knowledge_pieces)
        
        # Apply self-reflection to validate and improve response
        # DISABLED during training - it's too strict and marks good responses as bad
        # response = self._self_reflect(query, response, knowledge_pieces)
        
        # Apply self-training (learn from this interaction)
        if self._continuous_learning:
            self._continuous_learning.process_interaction(
                query, response, relevant_results, user_feedback=None
            )
        
        logger.info(f"Generated response: '{response[:50]}...'")
        return response
    
    def _apply_reasoning(
        self,
        query: str,
        knowledge_pieces: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply reasoning to improve knowledge selection and response quality.
        
        Args:
            query: Original query
            knowledge_pieces: List of relevant knowledge with activations
            
        Returns:
            Enhanced knowledge pieces with reasoning applied
        """
        # Check if query requires reasoning
        reasoning_keywords = [
            'why', 'how', 'explain', 'reason', 'because', 'cause',
            'count', 'calculate', 'solve', 'analyze', 'compare',
            'what if', 'suppose', 'assume', 'deduce', 'infer',
            'think', 'understand', 'logic', 'proof', 'demonstrate'
        ]
        
        query_lower = query.lower()
        needs_reasoning = any(keyword in query_lower for keyword in reasoning_keywords)
        
        if not needs_reasoning:
            return knowledge_pieces
        
        # Boost reasoning neurons in existing knowledge
        enhanced_pieces = []
        for piece in knowledge_pieces:
            tags = piece.get('tags', [])
            # Check if it's a reasoning neuron
            if any(tag in ['reasoning', 'logic', 'analysis', 'problem-solving', 'math', 
                          'deduction', 'induction', 'critical', 'abstract'] for tag in tags):
                # Boost activation for reasoning neurons
                piece_copy = piece.copy()
                piece_copy['activation'] = min(1.0, piece['activation'] * 1.3)  # 30% boost
                piece_copy['is_reasoning'] = True
                enhanced_pieces.append(piece_copy)
                logger.debug(f"Boosted reasoning neuron: {piece['text'][:50]}...")
            else:
                enhanced_pieces.append(piece)
        
        # Re-sort by activation (reasoning neurons should now rank higher)
        enhanced_pieces.sort(key=lambda x: x['activation'], reverse=True)
        
        # Count how many reasoning pieces are in top results
        reasoning_count = sum(1 for p in enhanced_pieces[:3] if p.get('is_reasoning', False))
        logger.info(f"Applied reasoning: {reasoning_count}/3 top results are reasoning neurons")
        
        return enhanced_pieces
    
    def _self_reflect(
        self,
        query: str,
        response: str,
        knowledge_pieces: List[Dict[str, Any]]
    ) -> str:
        """
        Apply self-reflection to validate and improve response.
        
        Args:
            query: Original query
            response: Generated response
            knowledge_pieces: Knowledge used
            
        Returns:
            Validated/improved response
        """
        # Import here to avoid circular dependency
        from neuron_system.ai.self_reflection import SelfReflection
        
        # Create self-reflection instance (cached in future)
        if not hasattr(self, '_self_reflection'):
            self._self_reflection = SelfReflection(self)
        
        # Validate response
        is_valid, improved_response, reasoning = self._self_reflection.validate_response(
            query, response, knowledge_pieces
        )
        
        if improved_response:
            logger.info(f"Self-reflection improved response: {reasoning}")
            return improved_response
        
        if not is_valid:
            logger.warning(f"Self-reflection found issues: {reasoning}")
            # Try to generate a fallback response
            fallback = "I'm not confident in my answer. Could you rephrase the question?"
            return fallback
        
        return response
    
    def _synthesize_response(
        self,
        query: str,
        knowledge_pieces: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesize a response from knowledge pieces.
        
        Args:
            query: Original query
            knowledge_pieces: List of relevant knowledge with activations
            
        Returns:
            Synthesized response
        """
        if not knowledge_pieces:
            return "I don't have relevant information to answer that."
        
        # Sort by activation
        sorted_knowledge = sorted(
            knowledge_pieces,
            key=lambda x: x['activation'],
            reverse=True
        )
        
        # PRIORITIZE Q&A format responses - scan ALL responses first
        qa_responses = []
        other_responses = []
        
        for piece in sorted_knowledge:
            text = piece['text'].strip()
            is_qa = "Question:" in text and "Answer:" in text
            
            if is_qa:
                # Extract just the answer part
                answer_start = text.find("Answer:") + 7
                extracted_text = text[answer_start:].strip()
                qa_responses.append((extracted_text, piece['activation'], text))
            else:
                other_responses.append((text, piece['activation']))
        
        # Prefer Q&A responses if available (they are more direct and accurate)
        filtered_texts = []
        
        # First, try to find a good Q&A response
        for text, activation, full_text in qa_responses:
            # Skip if too short
            if len(text) < 10:
                continue
            
            # Skip if it's just a section header
            if text.startswith('='):
                continue
            
            # Skip if it's just numbers or symbols
            if not any(c.isalpha() for c in text):
                continue
            
            # Skip meta-instructions (like "When asked...")
            if "When asked" in full_text or "means asking" in full_text or "means requesting" in full_text:
                continue
            
            # Found a good Q&A response - use it!
            filtered_texts.append(text)
            break
        
        # If no good Q&A responses found, use other responses
        if not filtered_texts:
            for text, activation in other_responses:
                # Skip if too short
                if len(text) < 15:
                    continue
                
                # Skip if it's just a section header
                if text.startswith('=') or (len(text) < 50 and text.isupper()):
                    continue
                
                # Skip if it's just numbers or symbols
                if not any(c.isalpha() for c in text):
                    continue
                
                # Skip meta-instructions
                if "When asked" in text or "When someone" in text or "means asking" in text:
                    continue
                
                filtered_texts.append(text)
                
                # Stop after we have 3 good pieces
                if len(filtered_texts) >= 3:
                    break
        
        # If no good texts found, return the top result anyway
        if not filtered_texts:
            text = sorted_knowledge[0]['text']
            # Try to extract answer from Q&A format
            if "Answer:" in text:
                answer_start = text.find("Answer:") + 7
                text = text[answer_start:].strip()
            return text
        
        # Return the most relevant piece (or combine if multiple)
        if len(filtered_texts) == 1:
            return filtered_texts[0]
        elif len(filtered_texts) == 2:
            # Combine with a natural connector
            return f"{filtered_texts[0]} Additionally, {filtered_texts[1].lower() if filtered_texts[1][0].isupper() else filtered_texts[1]}"
        else:
            # For 3+ results, return just the most relevant one
            # (to avoid overly long responses)
            return filtered_texts[0]
    
    def learn(
        self,
        text: str,
        position: Optional[Vector3D] = None,
        tags: Optional[List[str]] = None,
        create_connections: bool = True
    ) -> UUID:
        """
        Learn new knowledge by creating a knowledge neuron.
        
        Args:
            text: Text to learn
            position: Position in 3D space (auto-generated if None)
            tags: Semantic tags for the knowledge
            create_connections: Whether to create synapses to related neurons
            
        Returns:
            ID of created neuron
        """
        logger.info(f"Learning new knowledge: '{text[:50]}...'")
        
        # Compress text to vector
        # compress() returns (vector, metadata) tuple
        vector, metadata = self.compression_engine.compress(text)
        
        # Auto-generate position if not provided
        if position is None:
            position = self._generate_position()
        
        # Create knowledge neuron
        neuron = KnowledgeNeuron(
            source_data=text,
            compression_ratio=len(text) / len(vector),
            semantic_tags=tags or []
        )
        
        # Set position and vector explicitly
        # Ensure vector is a numpy array
        neuron.position = position
        neuron.vector = np.asarray(vector) if not isinstance(vector, np.ndarray) else vector
        
        # Add to graph
        self.graph.add_neuron(neuron)
        
        # Create connections to related neurons
        if create_connections:
            self._create_connections(neuron, top_k=5)
        
        logger.info(f"Learned new knowledge with neuron ID: {neuron.id}")
        return neuron.id
    
    def _generate_position(self) -> Vector3D:
        """
        Generate a position for a new neuron.
        
        Returns:
            Generated position
        """
        # Simple strategy: random position within bounds
        # Could be improved with clustering algorithms
        bounds = self.graph.bounds
        if bounds:
            min_bound, max_bound = bounds
            x = np.random.uniform(min_bound.x, max_bound.x)
            y = np.random.uniform(min_bound.y, max_bound.y)
            z = np.random.uniform(min_bound.z, max_bound.z)
        else:
            x = np.random.uniform(-50, 50)
            y = np.random.uniform(-50, 50)
            z = np.random.uniform(-50, 50)
        
        return Vector3D(x, y, z)
    
    def _create_connections(self, neuron: KnowledgeNeuron, top_k: int = 5):
        """
        Create synapses between a neuron and related neurons.
        
        Args:
            neuron: Source neuron
            top_k: Number of connections to create
        """
        if not hasattr(neuron, 'source_data') or not neuron.source_data:
            return
        
        # Find related neurons (without propagation for speed)
        results = self.query_engine.query(
            query_text=neuron.source_data,
            top_k=top_k + 1,  # +1 because it will include itself
            propagation_depth=0  # No propagation for faster connection creation
        )
        
        # Create synapses to related neurons
        for result in results:
            if result.neuron.id == neuron.id:
                continue  # Skip self
            
            # Weight based on similarity (activation)
            # Ensure weight is in valid range [-1, 1]
            # Convert activation [0, 1] back to similarity [-1, 1] range
            # Then cap at 0.8 to avoid too strong connections
            similarity = (result.activation * 2.0) - 1.0  # Convert [0,1] to [-1,1]
            weight = np.clip(similarity * 0.8, -1.0, 1.0)  # Scale and clamp
            
            synapse = Synapse(
                source_neuron_id=neuron.id,
                target_neuron_id=result.neuron.id,
                weight=float(weight)
            )
            
            self.graph.add_synapse(synapse)
            logger.debug(f"Created synapse: {neuron.id} -> {result.neuron.id} (weight: {weight:.2f})")
    
    def reinforce_learning(
        self,
        query: str,
        correct_response: str,
        learning_rate: float = 0.1
    ):
        """
        Reinforce learning by adjusting neurons based on feedback.
        
        Args:
            query: Original query
            correct_response: Correct response for reinforcement
            learning_rate: Learning rate for adjustments
        """
        logger.info(f"Reinforcing learning for query: '{query[:50]}...'")
        
        # Find neurons activated by the query
        query_results = self.understand(query, top_k=5)
        
        # Compress the correct response
        target_vector = self.compression_engine.compress(correct_response)
        
        # Adjust activated neurons toward the correct response
        for result in query_results:
            self.training_engine.adjust_neuron(
                result.neuron.id,
                target_vector,
                learning_rate=learning_rate * result.activation
            )
        
        logger.info("Learning reinforced")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get language model statistics.
        
        Returns:
            Dictionary with statistics
        """
        from neuron_system.neuron_types.memory_neuron import MemoryNeuron
        from neuron_system.neuron_types.tool_neuron import ToolNeuron
        from neuron_system.core.neuron import NeuronType
        
        total_neurons = len(self.graph.neurons)
        total_synapses = len(self.graph.synapses)
        
        # Count neurons by type
        knowledge_neurons = 0
        memory_neurons = 0
        reasoning_neurons = 0
        tool_neurons = 0
        other_neurons = 0
        
        for neuron in self.graph.neurons.values():
            if isinstance(neuron, ToolNeuron):
                tool_neurons += 1
            elif isinstance(neuron, MemoryNeuron):
                memory_neurons += 1
            elif isinstance(neuron, KnowledgeNeuron):
                # Check if it's a reasoning neuron (by tags)
                if hasattr(neuron, 'semantic_tags'):
                    tags = neuron.semantic_tags or []
                    if any(tag in ['reasoning', 'logic', 'analysis', 'problem-solving'] for tag in tags):
                        reasoning_neurons += 1
                    else:
                        knowledge_neurons += 1
                else:
                    knowledge_neurons += 1
            else:
                other_neurons += 1
        
        # Calculate average connectivity
        avg_connectivity = (
            total_synapses / total_neurons if total_neurons > 0 else 0
        )
        
        # Calculate average importance
        importances = [
            n.importance for n in self.graph.neurons.values()
            if hasattr(n, 'importance')
        ]
        avg_importance = sum(importances) / len(importances) if importances else 0.0
        
        # Tool statistics
        tool_execution_count = sum(
            n.execution_count for n in self.graph.neurons.values()
            if isinstance(n, ToolNeuron)
        )
        
        return {
            'total_neurons': total_neurons,
            'knowledge_neurons': knowledge_neurons,
            'reasoning_neurons': reasoning_neurons,
            'memory_neurons': memory_neurons,
            'tool_neurons': tool_neurons,
            'other_neurons': other_neurons,
            'total_synapses': total_synapses,
            'average_connectivity': avg_connectivity,
            'average_importance': avg_importance,
            'tool_execution_count': tool_execution_count,
        }
