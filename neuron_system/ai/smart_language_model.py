"""
Smart Language Model - Nutzt Neural Inference Engine für intelligente Antworten.

Erweitert das bestehende LanguageModel mit echter KI-Inferenz-Logik.
"""

import logging
import numpy as np
from typing import List, Optional, Dict, Any
from uuid import UUID

from neuron_system.ai.language_model import LanguageModel
from neuron_system.ai.neural_inference import NeuralInferenceEngine
from neuron_system.engines.query import ActivatedNeuron

logger = logging.getLogger(__name__)


class SmartLanguageModel(LanguageModel):
    """
    Intelligentes Language Model mit Neural Inference Engine.
    
    Nutzt echte Transformer-Logik für bessere Antworten.
    """
    
    def __init__(
        self,
        graph,
        compression_engine,
        query_engine,
        training_engine,
        enable_self_training: bool = True,
        pretrained_model: str = "distilbert-base-uncased"
    ):
        """
        Initialize Smart Language Model.
        
        Args:
            graph: Neuron graph
            compression_engine: Compression engine
            query_engine: Query engine
            training_engine: Training engine
            enable_self_training: Enable self-training
            pretrained_model: Hugging Face model to load
        """
        # Initialize parent
        super().__init__(
            graph, compression_engine, query_engine, 
            training_engine, enable_self_training
        )
        
        # Initialize Neural Inference Engine (dimensions auto-detected)
        self.neural_engine = NeuralInferenceEngine(
            embedding_dim=None,  # Auto-detect from model
            num_attention_heads=None,  # Auto-calculate
            hidden_dim=None  # Auto-calculate
        )
        
        # Load pretrained weights
        try:
            logger.info(f"Loading pretrained model: {pretrained_model}")
            self.neural_engine.initialize_from_pretrained(pretrained_model)
            
            # Validate dimension compatibility with compression engine
            # Trigger model load to get actual dimension
            self.compression_engine._ensure_model_loaded()
            compression_dim = self.compression_engine.vector_dim
            neural_dim = self.neural_engine.embedding_dim
            
            if compression_dim != neural_dim:
                logger.warning(
                    f"Dimension mismatch: Compression={compression_dim}D, Neural={neural_dim}D. "
                    f"Adjusting neural engine to {compression_dim}D"
                )
                # Re-initialize with correct dimension
                self.neural_engine = NeuralInferenceEngine(
                    embedding_dim=compression_dim,
                    num_attention_heads=None,
                    hidden_dim=None
                )
                self.neural_engine.initialize_from_pretrained(pretrained_model)
            
            self.neural_enabled = True
            logger.info(
                f"Neural Inference Engine ready: {self.neural_engine.embedding_dim}D, "
                f"{self.neural_engine.num_attention_heads} heads, "
                f"{self.neural_engine.hidden_dim} hidden"
            )
        except Exception as e:
            logger.warning(f"Could not load pretrained model: {e}")
            logger.warning("Falling back to standard inference")
            self.neural_enabled = False
    
    def generate_response(
        self,
        query: str,
        context_size: int = 10,
        min_activation: float = 0.2,
        propagation_depth: int = 0,
        use_reasoning: bool = True,
        use_neural_inference: bool = True
    ) -> str:
        """
        Generate response mit Neural Inference Engine.
        
        Args:
            query: Input query
            context_size: Number of neurons to consider
            min_activation: Minimum activation threshold
            propagation_depth: Depth of propagation
            use_reasoning: Use reasoning neurons
            use_neural_inference: Use neural inference engine
            
        Returns:
            Generated response
        """
        logger.info(f"Generating response for: '{query[:50]}...'")
        
        # Find relevant neurons (mehr als vorher für bessere Auswahl)
        initial_results = self.understand(
            query, 
            top_k=context_size * 2,  # Hole mehr Kandidaten
            propagation_depth=propagation_depth
        )
        
        if not initial_results:
            return "I don't have enough knowledge to answer that question."
        
        # === NEURAL INFERENCE ===
        if use_neural_inference and self.neural_enabled:
            results = self._apply_neural_inference(query, initial_results, context_size)
        else:
            # Fallback: Standard filtering
            results = [r for r in initial_results if r.activation >= min_activation][:context_size]
        
        if not results:
            return "I'm not confident enough to answer that question."
        
        # Extract knowledge
        knowledge_pieces = []
        for result in results:
            neuron = result.neuron
            if hasattr(neuron, 'source_data') and neuron.source_data:
                knowledge_pieces.append({
                    'text': neuron.source_data,
                    'activation': result.activation,
                    'tags': getattr(neuron, 'semantic_tags', []),
                    'importance': getattr(neuron, 'importance', 0.5)
                })
        
        # Apply reasoning if enabled
        if use_reasoning:
            knowledge_pieces = self._apply_reasoning(query, knowledge_pieces)
        
        # Synthesize response
        response = self._synthesize_response(query, knowledge_pieces)
        
        # Apply self-training
        if self._continuous_learning:
            self._continuous_learning.process_interaction(
                query, response, results, user_feedback=None
            )
        
        logger.info(f"Generated response: '{response[:50]}...'")
        return response
    
    def _apply_neural_inference(
        self,
        query: str,
        candidates: List[ActivatedNeuron],
        top_k: int
    ) -> List[ActivatedNeuron]:
        """
        Nutze Neural Inference Engine für intelligente Neuron-Auswahl.
        
        Args:
            query: Query text
            candidates: Candidate neurons
            top_k: Number to select
            
        Returns:
            Re-ranked neurons mit Neural Inference Scores
        """
        logger.debug("Applying neural inference...")
        
        # Compress query
        query_vector, _ = self.compression_engine.compress(query)
        query_vector = np.asarray(query_vector)
        
        # Extract neuron vectors and metadata
        neuron_vectors = []
        neuron_metadata = []
        
        for candidate in candidates:
            if candidate.neuron.vector is not None:
                neuron_vectors.append(candidate.neuron.vector)
                neuron_metadata.append({
                    'tags': getattr(candidate.neuron, 'semantic_tags', []),
                    'importance': getattr(candidate.neuron, 'importance', 0.5)
                })
        
        if not neuron_vectors:
            return candidates[:top_k]
        
        neuron_vectors = np.array(neuron_vectors)
        
        # Compute neural reasoning scores
        reasoning_scores = self.neural_engine.compute_reasoning_score(
            query_vector, neuron_vectors, neuron_metadata
        )
        
        # Kombiniere mit ursprünglichen Activation-Scores
        # 60% Neural Inference, 40% Original Similarity
        for i, candidate in enumerate(candidates[:len(reasoning_scores)]):
            original_score = float(candidate.activation)
            neural_score = float(reasoning_scores[i])
            
            # Kombinierter Score
            combined_score = 0.6 * neural_score + 0.4 * original_score
            
            # Update activation
            candidate.activation = combined_score
            
            logger.debug(
                f"Neuron {candidate.neuron.id}: "
                f"original={original_score:.3f}, neural={neural_score:.3f}, "
                f"combined={combined_score:.3f}"
            )
        
        # Re-sort by combined score
        candidates.sort(key=lambda x: x.activation, reverse=True)
        
        logger.debug(f"Neural inference complete. Top score: {candidates[0].activation:.3f}")
        
        return candidates[:top_k]
    
    def _synthesize_response(
        self,
        query: str,
        knowledge_pieces: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesize response mit Neural Context Understanding.
        
        Überschreibt die Parent-Methode mit intelligenterer Synthese.
        """
        if not knowledge_pieces:
            return "I don't have relevant information to answer that."
        
        # Sort by combined score (activation * importance)
        sorted_knowledge = sorted(
            knowledge_pieces,
            key=lambda x: x['activation'] * x.get('importance', 0.5),
            reverse=True
        )
        
        # === INTELLIGENTE RESPONSE-AUSWAHL ===
        
        # 1. Prioritize Q&A format responses
        qa_responses = []
        other_responses = []
        
        for piece in sorted_knowledge:
            text = piece['text'].strip()
            is_qa = "Question:" in text and "Answer:" in text
            
            if is_qa:
                # Extract answer
                answer_start = text.find("Answer:") + 7
                extracted_text = text[answer_start:].strip()
                qa_responses.append((
                    extracted_text, 
                    piece['activation'], 
                    piece.get('importance', 0.5),
                    text
                ))
            else:
                other_responses.append((
                    text, 
                    piece['activation'],
                    piece.get('importance', 0.5)
                ))
        
        # 2. Filter and select best response
        filtered_texts = []
        
        # Try Q&A responses first
        for text, activation, importance, full_text in qa_responses:
            if len(text) < 10:
                continue
            if text.startswith('='):
                continue
            if not any(c.isalpha() for c in text):
                continue
            if "When asked" in full_text or "means asking" in full_text:
                continue
            
            # Good Q&A response found
            filtered_texts.append(text)
            break
        
        # If no good Q&A, use other responses
        if not filtered_texts:
            for text, activation, importance in other_responses:
                if len(text) < 15:
                    continue
                if text.startswith('=') or (len(text) < 50 and text.isupper()):
                    continue
                if not any(c.isalpha() for c in text):
                    continue
                if "When asked" in text or "means asking" in text:
                    continue
                
                filtered_texts.append(text)
                
                if len(filtered_texts) >= 2:
                    break
        
        # Fallback: use top result anyway
        if not filtered_texts:
            text = sorted_knowledge[0]['text']
            if "Answer:" in text:
                answer_start = text.find("Answer:") + 7
                text = text[answer_start:].strip()
            return text
        
        # === NEURAL CONTEXT COMBINATION ===
        # Wenn mehrere gute Antworten, nutze Neural Engine für Kombination
        if len(filtered_texts) > 1 and self.neural_enabled:
            return self._neural_combine_responses(query, filtered_texts)
        
        # Single response
        return filtered_texts[0]
    
    def _neural_combine_responses(
        self,
        query: str,
        responses: List[str]
    ) -> str:
        """
        Kombiniere mehrere Responses intelligent mit Neural Engine.
        
        Args:
            query: Original query
            responses: List of response candidates
            
        Returns:
            Best combined response
        """
        # Für jetzt: Nimm die beste einzelne Response
        # TODO: Implementiere echte Response-Kombination mit Attention
        
        # Compress responses
        response_vectors = []
        for response in responses[:3]:  # Max 3
            vec, _ = self.compression_engine.compress(response)
            response_vectors.append(vec)
        
        query_vec, _ = self.compression_engine.compress(query)
        
        # Compute attention scores
        response_vectors = np.array(response_vectors)
        query_vec = np.array(query_vec).reshape(1, -1)
        
        # Simple similarity for now
        similarities = np.dot(response_vectors, query_vec.T).flatten()
        best_idx = np.argmax(similarities)
        
        return responses[best_idx]
    
    def get_neural_statistics(self) -> Dict[str, Any]:
        """Get statistics about neural inference."""
        stats = self.get_statistics()
        
        stats['neural_inference_enabled'] = self.neural_enabled
        
        if self.neural_enabled:
            stats['neural_engine_info'] = self.neural_engine.get_model_info()
        
        return stats
