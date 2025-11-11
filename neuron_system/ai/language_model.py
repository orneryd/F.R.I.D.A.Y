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
        enable_self_training: bool = True,
        enable_logic_engine: bool = True
    ):
        """
        Initialize the language model.
        
        Args:
            graph: Neuron graph instance
            compression_engine: Compression engine for text encoding
            query_engine: Query engine for retrieval
            training_engine: Training engine for learning
            enable_self_training: Enable continuous self-training
            enable_logic_engine: Enable logic execution from assimilated models
        """
        self.graph = graph
        self.compression_engine = compression_engine
        self.query_engine = query_engine
        self.training_engine = training_engine
        
        # Initialize logic engine for executing assimilated logic
        if enable_logic_engine:
            from neuron_system.ai.logic_engine import LogicEngine
            self._logic_engine = LogicEngine(graph, query_engine)
            stats = self._logic_engine.get_logic_stats()
            if stats['total_rules'] > 0:
                logger.info(f"Logic engine initialized with {stats['total_rules']} rules")
            else:
                logger.info("Logic engine initialized (no rules yet - assimilate models to add logic)")
        else:
            self._logic_engine = None
        
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
        use_reasoning: bool = True,
        use_generative: bool = True,
        use_chain_of_thought: bool = True,
        use_self_reflection: bool = True,
        store_reasoning_trace: bool = True,
        output_think_tags: bool = True
    ) -> str:
        """
        Generate a response to a query using activated neurons.
        
        Args:
            query: Input query
            context_size: Number of neurons to use for context
            min_activation: Minimum activation threshold
            propagation_depth: Depth of activation propagation (0 = no propagation)
            use_reasoning: Whether to use reasoning neurons to improve response
            use_generative: Whether to use generative model (learns and creates new responses)
            use_chain_of_thought: Whether to create a lightweight CoT reasoning trace
            use_self_reflection: Whether to validate/improve with self-reflection
            store_reasoning_trace: Whether to persist the reasoning trace as memory
            output_think_tags: Whether to include <think>...</think> CoT block before final answer
            
        Returns:
            Generated response text
        """
        logger.info(f"Generating response for: '{query[:50]}...'")
        
        # DISABLED: ImprovedGenerator - we use GenerativeSynthesizer instead
        # The ImprovedGenerator only returns stored text, not generated responses
        
        # Original method (fallback)
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
        used_neuron_ids = []
        for result in relevant_results[:context_size]:
            neuron = result.neuron
            if hasattr(neuron, 'source_data') and neuron.source_data:
                knowledge_pieces.append({
                    'text': neuron.source_data,
                    'activation': result.activation,
                    'tags': getattr(neuron, 'semantic_tags', [])
                })
                used_neuron_ids.append(str(getattr(neuron, "id", "")))
        
        # Optional: build a lightweight Chain-of-Thought trace (no recursion)
        reasoning_steps = []
        if use_chain_of_thought:
            ql = (query or "").lower()
            # Step 1: restate the problem
            reasoning_steps.append(f"Understand: {query}")
            # Step 2: classify the question intent
            if any(w in ql for w in ['why', 'because', 'reason']):
                reasoning_steps.append("Type: causal explanation")
            elif any(w in ql for w in ['how', 'process', 'method', 'steps']):
                reasoning_steps.append("Type: procedural steps")
            elif any(w in ql for w in ['compare', 'difference', 'similar']):
                reasoning_steps.append("Type: comparison/contrast")
            elif any(w in ql for w in ['what', 'define', 'explain']):
                reasoning_steps.append("Type: definitional/explanatory")
            else:
                reasoning_steps.append("Type: general informative")
            # Step 3: evidence plan
            reasoning_steps.append("Plan: select top relevant knowledge, check consistency, synthesize concise answer")
        
        # Apply reasoning if enabled
        if use_reasoning:
            knowledge_pieces = self._apply_reasoning(query, knowledge_pieces)
        
        # Generate response based on activated knowledge
        # Try logic engine first if available (uses assimilated logic)
        if self._logic_engine and self._logic_engine.get_logic_stats()['total_rules'] > 0:
            try:
                response = self._logic_engine.generate_with_logic(query, knowledge_pieces)
                if response and len(response) > 10:
                    logger.info("Using logic engine (assimilated logic)")
                else:
                    response = self._synthesize_response(query, knowledge_pieces)
            except Exception as e:
                logger.debug(f"Logic engine failed, using fallback: {e}")
                response = self._synthesize_response(query, knowledge_pieces)
        else:
            response = self._synthesize_response(query, knowledge_pieces)
        
        # Apply self-reflection to validate and improve response (lenient)
        if use_self_reflection:
            response = self._self_reflect(query, response, knowledge_pieces)
        
        # Format output with <think>...</think> if requested
        if output_think_tags and use_chain_of_thought:
            try:
                think_block = ""
                if reasoning_steps:
                    # Join steps into a single block; keep short and readable
                    think_block = "<think>\n" + "\n".join(reasoning_steps) + "\n</think>\n"
                if think_block:
                    response = f"{think_block}{response}"
            except Exception as e:
                logger.debug(f"Formatting think tags failed: {e}")
        
        # Apply self-training (learn from this interaction)
        if self._continuous_learning:
            self._continuous_learning.process_interaction(
                query, response, relevant_results, user_feedback=None
            )
        
        # Optionally store reasoning trace as a memory for transparency/audit
        if store_reasoning_trace and (use_chain_of_thought or use_self_reflection):
            try:
                from neuron_system.neuron_types.memory_neuron import MemoryManager
                if not hasattr(self, '_memory_manager'):
                    self._memory_manager = MemoryManager(self.graph, self.compression_engine)
                
                trace_lines = []
                if reasoning_steps:
                    trace_lines.append("ChainOfThought:")
                    trace_lines.extend([f"- {s}" for s in reasoning_steps])
                trace_lines.append("Answer: " + response)
                trace_lines.append("UsedNeurons: " + ", ".join(used_neuron_ids[:10]))
                
                trace_text = "\n".join(trace_lines)
                self._memory_manager.create_memory(
                    content=trace_text,
                    memory_type="episodic",
                    context={
                        "query": query,
                        "used_neurons": used_neuron_ids,
                    },
                    importance=0.6
                )
            except Exception as e:
                logger.debug(f"Storing reasoning trace failed: {e}")
        
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
        Synthesize a response from knowledge pieces using intelligent synthesis.
        
        Args:
            query: Original query
            knowledge_pieces: List of relevant knowledge with activations
            
        Returns:
            Synthesized response
        """
        if not knowledge_pieces:
            return "I don't have relevant information to answer that."
        
        # Try generative synthesis first (creates original sentences)
        try:
            from neuron_system.ai.generative_synthesis import GenerativeSynthesizer
            
            if not hasattr(self, '_generative_synthesizer'):
                self._generative_synthesizer = GenerativeSynthesizer(model_type="auto")
            
            response = self._generative_synthesizer.synthesize_response(
                query, knowledge_pieces, max_length=150
            )
            
            if response and len(response) > 10:
                logger.info("Using generative synthesis")
                return response
            
        except Exception as e:
            logger.warning(f"Generative synthesis failed: {e}")
        
        # Fallback to intelligent synthesizer
        try:
            from neuron_system.ai.intelligent_synthesis import IntelligentSynthesizer
            
            if not hasattr(self, '_synthesizer'):
                self._synthesizer = IntelligentSynthesizer()
            
            response = self._synthesizer.synthesize_response(
                query, knowledge_pieces, min_confidence=0.3
            )
            
            return response
            
        except Exception as e:
            logger.warning(f"Intelligent synthesis failed, using fallback: {e}")
            # Fallback to simple synthesis
            return self._simple_synthesis(query, knowledge_pieces)
    
    def _simple_synthesis(
        self,
        query: str,
        knowledge_pieces: List[Dict[str, Any]]
    ) -> str:
        """
        Simple fallback synthesis method.
        
        WICHTIG: Nimmt NUR die beste Antwort, kombiniert NICHTS!
        Alle Überlegungen gehören ins Reasoning, nicht in die finale Antwort.
        
        Args:
            query: Original query
            knowledge_pieces: List of relevant knowledge with activations
            
        Returns:
            Single, clean synthesized response
        """
        # Sort by activation
        sorted_knowledge = sorted(
            knowledge_pieces,
            key=lambda x: x['activation'],
            reverse=True
        )
        
        # Extract ONLY the best answer - NO COMBINATION
        best_answer = None
        
        for piece in sorted_knowledge:
            text = piece['text'].strip()
            
            # Try Q&A format - extract only the answer part
            if "Answer:" in text:
                answer_start = text.find("Answer:") + 7
                answer = text[answer_start:].strip()
                
                if len(answer) > 10:
                    best_answer = answer
                    break
            
            # Use full text if good enough
            if len(text) > 15 and any(c.isalpha() for c in text):
                best_answer = text
                break
        
        # If no answer found, use first piece
        if not best_answer:
            best_answer = sorted_knowledge[0]['text']
        
        # ALWAYS clean the final answer before returning
        return self._clean_final_answer(best_answer)
    
    def _clean_final_answer(self, answer: str) -> str:
        """
        Bereinigt die finale Antwort von Meta-Kommentaren und Zusätzen.
        
        Entfernt:
        - "Zusätzlich:", "Additionally:", etc.
        - "Based on", "According to", etc.
        - Mehrfache Sätze die das gleiche sagen
        - Technische Artefakte
        
        Args:
            answer: Rohe Antwort
            
        Returns:
            Bereinigte, fokussierte Antwort
        """
        import re
        
        # Entferne Meta-Phrasen am Anfang
        meta_phrases = [
            r'^Based on[^,\.]+[,\.]?\s*',
            r'^According to[^,\.]+[,\.]?\s*',
            r'^In my (opinion|view|analysis)[,\.]?\s*',
            r'^I (think|believe|found) (that\s+)?',
            r'^As (mentioned|noted|stated)[^,\.]+[,\.]?\s*',
        ]
        
        for pattern in meta_phrases:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
        
        # Entferne "Zusätzlich:" und ähnliche Marker
        # Das ist das Hauptproblem!
        # WICHTIG: Auch mit Encoding-Problemen (ä vs õ)
        answer = re.sub(r'\.\s*Zus[aäõ]tzlich:\s*', '. ', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\.\s*Additionally:\s*', '. ', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\.\s*Furthermore:\s*', '. ', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\.\s*Moreover:\s*', '. ', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\.\s*Also:\s*', '. ', answer, flags=re.IGNORECASE)
        
        # Wenn nach "Zusätzlich:" ein zweiter Satz kommt, nimm NUR den ersten
        # Beispiel: "Satz 1. Zusätzlich: Satz 2" -> "Satz 1"
        if re.search(r'\.\s*(Zus[aäõ]tzlich|Additionally|Furthermore|Moreover|Also):', answer, re.IGNORECASE):
            # Nimm nur den Teil vor dem Marker
            parts = re.split(r'\.\s*(Zus[aäõ]tzlich|Additionally|Furthermore|Moreover|Also):', answer, maxsplit=1, flags=re.IGNORECASE)
            if parts and len(parts[0]) > 10:
                answer = parts[0] + '.'
        
        # Entferne doppelte Punkte
        answer = re.sub(r'\.\.+', '.', answer)
        
        # Entferne technische Artefakte
        answer = re.sub(r'@-@', '', answer)
        answer = re.sub(r'< unk >', '', answer)
        
        # Entferne mehrfache Leerzeichen
        answer = re.sub(r'\s+', ' ', answer)
        
        # Trim
        answer = answer.strip()
        
        # Stelle sicher dass Satz mit Punkt endet
        if answer and not answer[-1] in '.!?':
            answer += '.'
        
        return answer
    
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
            position = self._generate_position(tags=tags)
        
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
    
    def learn_batch(
        self,
        texts: List[str],
        tags_list: Optional[List[List[str]]] = None,
        create_connections: bool = False
    ) -> List[UUID]:
        """
        Learn multiple knowledge pieces in batch (much faster).
        
        Args:
            texts: List of texts to learn
            tags_list: List of tag lists for each text
            create_connections: Whether to create synapses
            
        Returns:
            List of created neuron IDs
        """
        logger.info(f"Batch learning {len(texts)} knowledge pieces...")
        
        if tags_list is None:
            tags_list = [None] * len(texts)
        
        neuron_ids = []
        neurons_to_add = []
        
        # Create all neurons first (without saving)
        for text, tags in zip(texts, tags_list):
            # Compress text to vector
            vector, metadata = self.compression_engine.compress(text)
            
            # Auto-generate position
            position = self._generate_position(tags=tags)
            
            # Create knowledge neuron
            neuron = KnowledgeNeuron(
                source_data=text,
                compression_ratio=len(text) / len(vector),
                semantic_tags=tags or []
            )
            
            neuron.position = position
            neuron.vector = np.asarray(vector) if not isinstance(vector, np.ndarray) else vector
            
            neurons_to_add.append(neuron)
            neuron_ids.append(neuron.id)
        
        # Add all neurons at once
        for neuron in neurons_to_add:
            self.graph.add_neuron(neuron)
        
        # Save once at the end
        self.graph.save()
        
        logger.info(f"Batch learned {len(neuron_ids)} neurons")
        return neuron_ids
    
    def _generate_position(self, tags: Optional[List[str]] = None) -> Vector3D:
        """
        Generate a position for a new neuron.
        
        Uses smart positioning if available, otherwise random.
        
        Args:
            tags: Semantic tags for topic-based positioning
        
        Returns:
            Generated position
        """
        # Try smart positioning if available
        try:
            from neuron_system.spatial.smart_positioning import SmartPositioner
            
            if not hasattr(self, '_smart_positioner'):
                self._smart_positioner = SmartPositioner(self.graph.bounds)
            
            # Use topic-based positioning if tags provided
            if tags:
                return self._smart_positioner.position_by_topic(tags, spread=10.0)
            
            # Otherwise semantic positioning
            existing = list(self.graph.neurons.values())
            if existing:
                # Create temporary neuron for similarity check
                # (we don't have the neuron object yet)
                return self._smart_positioner.position_density_aware(
                    existing, min_distance=5.0
                )
        except Exception as e:
            logger.debug(f"Smart positioning failed, using random: {e}")
        
        # Fallback: random position within bounds
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
