"""
Neural Inference Engine - Echte KI-Logik für intelligente Datenverarbeitung.

Dieser Engine implementiert echte neuronale Netzwerk-Logik:
- Multi-Layer Attention (wie Transformer)
- Kontextuelles Reasoning
- Intelligente Aktivierungsmuster
- Dynamische Gewichtung

Nutzt die bestehenden Neuronen, aber mit echter KI-Inferenz statt nur Vektor-Suche.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AttentionResult:
    """Ergebnis eines Attention-Layers."""
    attended_vectors: np.ndarray  # Gewichtete Vektoren
    attention_weights: np.ndarray  # Attention-Gewichte
    context_vector: np.ndarray  # Finaler Kontext-Vektor


class NeuralInferenceEngine:
    """
    Echter Neural Inference Engine mit Transformer-ähnlicher Logik.
    
    Implementiert:
    - Multi-Head Attention
    - Feed-Forward Networks
    - Layer Normalization
    - Residual Connections
    """
    
    def __init__(
        self,
        embedding_dim: int = None,  # Auto-detect if None
        num_attention_heads: int = None,  # Auto-calculate if None
        hidden_dim: int = None,  # Auto-calculate if None
        dropout_rate: float = 0.1
    ):
        """
        Initialize Neural Inference Engine.
        
        Args:
            embedding_dim: Dimension der Neuron-Vektoren (None = auto-detect)
            num_attention_heads: Anzahl der Attention-Heads (None = auto-calculate)
            hidden_dim: Dimension des Hidden Layers (None = auto-calculate)
            dropout_rate: Dropout-Rate für Regularisierung
        """
        self.embedding_dim = embedding_dim  # Will be set from model
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Attention-Parameter (werden aus Hugging Face Modell geladen)
        self.query_weights = None
        self.key_weights = None
        self.value_weights = None
        self.output_weights = None
        
        # Feed-Forward Parameter
        self.ff_weights_1 = None
        self.ff_bias_1 = None
        self.ff_weights_2 = None
        self.ff_bias_2 = None
        
        # Layer Normalization Parameter
        self.ln_gamma = None
        self.ln_beta = None
        
        logger.info(
            f"Neural Inference Engine initialized: "
            f"dim={embedding_dim}, heads={num_attention_heads}, hidden={hidden_dim}"
        )
    
    def initialize_from_pretrained(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialisiere Gewichte von einem vortrainierten Hugging Face Modell.
        
        Args:
            model_name: Name des Hugging Face Modells
        """
        try:
            from transformers import AutoModel
            import torch
            
            logger.info(f"Loading pretrained model: {model_name}")
            model = AutoModel.from_pretrained(model_name)
            
            # Extrahiere Gewichte vom ersten Transformer-Layer
            first_layer = model.transformer.layer[0] if hasattr(model, 'transformer') else model.encoder.layer[0]
            
            # Attention-Gewichte
            attention = first_layer.attention
            q_weights = attention.q_lin.weight.detach().numpy()
            k_weights = attention.k_lin.weight.detach().numpy()
            v_weights = attention.v_lin.weight.detach().numpy()
            o_weights = attention.out_lin.weight.detach().numpy()
            
            # Model dimension (z.B. 768 für BERT/DistilBERT)
            model_dim = q_weights.shape[0]
            
            # Auto-detect embedding_dim if not set
            if self.embedding_dim is None:
                self.embedding_dim = model_dim
                logger.info(f"Auto-detected embedding dimension: {self.embedding_dim}")
            
            # Auto-calculate num_attention_heads if not set
            if self.num_attention_heads is None:
                # Standard: 12 heads for 768D, 8 for 512D, 6 for 384D
                if self.embedding_dim >= 768:
                    self.num_attention_heads = 12
                elif self.embedding_dim >= 512:
                    self.num_attention_heads = 8
                else:
                    self.num_attention_heads = 6
                logger.info(f"Auto-calculated attention heads: {self.num_attention_heads}")
            
            # Auto-calculate hidden_dim if not set
            if self.hidden_dim is None:
                # Standard: 4x embedding_dim (wie in BERT)
                self.hidden_dim = self.embedding_dim * 4
                logger.info(f"Auto-calculated hidden dimension: {self.hidden_dim}")
            
            logger.info(f"Model dimension: {model_dim}, Target dimension: {self.embedding_dim}")
            
            # Wenn Dimensionen nicht übereinstimmen, projiziere
            if model_dim != self.embedding_dim:
                logger.info(f"Projecting weights from {model_dim} to {self.embedding_dim}")
                
                # Einfache Projektion: Nimm die ersten N Dimensionen
                # (Alternativ: PCA oder trainierte Projektion)
                self.query_weights = q_weights[:self.embedding_dim, :self.embedding_dim]
                self.key_weights = k_weights[:self.embedding_dim, :self.embedding_dim]
                self.value_weights = v_weights[:self.embedding_dim, :self.embedding_dim]
                self.output_weights = o_weights[:self.embedding_dim, :self.embedding_dim]
            else:
                self.query_weights = q_weights
                self.key_weights = k_weights
                self.value_weights = v_weights
                self.output_weights = o_weights
            
            # Feed-Forward-Gewichte
            ffn = first_layer.ffn
            ff1_weights = ffn.lin1.weight.detach().numpy()
            ff1_bias = ffn.lin1.bias.detach().numpy()
            ff2_weights = ffn.lin2.weight.detach().numpy()
            ff2_bias = ffn.lin2.bias.detach().numpy()
            
            # Projiziere Feed-Forward Gewichte
            if model_dim != self.embedding_dim:
                # Input projection: model_dim -> embedding_dim
                self.ff_weights_1 = ff1_weights[:self.hidden_dim, :self.embedding_dim]
                self.ff_bias_1 = ff1_bias[:self.hidden_dim]
                
                # Output projection: hidden_dim -> embedding_dim
                self.ff_weights_2 = ff2_weights[:self.embedding_dim, :self.hidden_dim]
                self.ff_bias_2 = ff2_bias[:self.embedding_dim]
            else:
                self.ff_weights_1 = ff1_weights
                self.ff_bias_1 = ff1_bias
                self.ff_weights_2 = ff2_weights
                self.ff_bias_2 = ff2_bias
            
            # Layer Normalization
            ln_gamma = first_layer.sa_layer_norm.weight.detach().numpy()
            ln_beta = first_layer.sa_layer_norm.bias.detach().numpy()
            
            if model_dim != self.embedding_dim:
                self.ln_gamma = ln_gamma[:self.embedding_dim]
                self.ln_beta = ln_beta[:self.embedding_dim]
            else:
                self.ln_gamma = ln_gamma
                self.ln_beta = ln_beta
            
            logger.info("✓ Successfully loaded pretrained weights")
            logger.info(f"  Query weights shape: {self.query_weights.shape}")
            logger.info(f"  Key weights shape: {self.key_weights.shape}")
            logger.info(f"  Value weights shape: {self.value_weights.shape}")
            logger.info(f"  FF weights shape: {self.ff_weights_1.shape}")
            
        except ImportError:
            logger.error("transformers library not installed. Run: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise
    
    def multi_head_attention(
        self,
        query_vectors: np.ndarray,
        key_vectors: np.ndarray,
        value_vectors: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> AttentionResult:
        """
        Multi-Head Attention Mechanismus (wie in Transformers).
        
        Args:
            query_vectors: Query-Vektoren [batch_size, seq_len, dim]
            key_vectors: Key-Vektoren [batch_size, seq_len, dim]
            value_vectors: Value-Vektoren [batch_size, seq_len, dim]
            mask: Optional attention mask
            
        Returns:
            AttentionResult mit gewichteten Vektoren
        """
        if self.query_weights is None:
            raise ValueError("Model not initialized. Call initialize_from_pretrained() first")
        
        batch_size, seq_len, dim = query_vectors.shape
        head_dim = dim // self.num_attention_heads
        
        # Linear projections für Q, K, V
        # Ensure weights are correct shape [dim, dim]
        Q = np.dot(query_vectors, self.query_weights.T)  # [batch, seq, dim]
        K = np.dot(key_vectors, self.key_weights.T)
        V = np.dot(value_vectors, self.value_weights.T)
        
        # Verify dimensions
        assert Q.shape == (batch_size, seq_len, dim), f"Q shape mismatch: {Q.shape} vs expected ({batch_size}, {seq_len}, {dim})"
        
        # Reshape für Multi-Head: [batch, heads, seq, head_dim]
        Q = Q.reshape(batch_size, seq_len, self.num_attention_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_attention_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_attention_heads, head_dim).transpose(0, 2, 1, 3)
        
        # Scaled Dot-Product Attention
        # scores = Q @ K^T / sqrt(head_dim)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Softmax über die Key-Dimension
        attention_weights = self._softmax(scores, axis=-1)
        
        # Apply dropout (nur während Training)
        # attention_weights = self._dropout(attention_weights, self.dropout_rate)
        
        # Gewichtete Summe der Values
        attended = np.matmul(attention_weights, V)  # [batch, heads, seq, head_dim]
        
        # Concatenate heads: [batch, seq, dim]
        attended = attended.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, dim)
        
        # Output projection
        output = np.dot(attended, self.output_weights.T)
        
        # Context vector: Durchschnitt über Sequenz
        context_vector = np.mean(output, axis=1)  # [batch, dim]
        
        return AttentionResult(
            attended_vectors=output,
            attention_weights=attention_weights.mean(axis=1),  # Average über Heads
            context_vector=context_vector
        )
    
    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Feed-Forward Network (2-Layer MLP mit GELU).
        
        Args:
            x: Input [batch, seq, dim]
            
        Returns:
            Output [batch, seq, dim]
        """
        if self.ff_weights_1 is None:
            raise ValueError("Model not initialized")
        
        # First layer: dim -> hidden_dim
        hidden = np.dot(x, self.ff_weights_1.T) + self.ff_bias_1
        hidden = self._gelu(hidden)
        
        # Second layer: hidden_dim -> dim
        output = np.dot(hidden, self.ff_weights_2.T) + self.ff_bias_2
        
        return output
    
    def layer_norm(self, x: np.ndarray) -> np.ndarray:
        """
        Layer Normalization.
        
        Args:
            x: Input [batch, seq, dim]
            
        Returns:
            Normalized output
        """
        if self.ln_gamma is None:
            # Fallback: Standard normalization
            mean = np.mean(x, axis=-1, keepdims=True)
            std = np.std(x, axis=-1, keepdims=True)
            return (x - mean) / (std + 1e-5)
        
        # Mit gelernten Parametern
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        normalized = (x - mean) / (std + 1e-5)
        
        return self.ln_gamma * normalized + self.ln_beta
    
    def forward_pass(
        self,
        neuron_vectors: np.ndarray,
        query_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vollständiger Forward Pass durch das Neural Network.
        
        Args:
            neuron_vectors: Vektoren der aktivierten Neuronen [num_neurons, dim]
            query_vector: Query-Vektor [dim]
            
        Returns:
            Tuple von (context_vector, attention_weights)
        """
        # Reshape für Batch-Processing
        batch_size = 1
        num_neurons = neuron_vectors.shape[0]
        
        # Query als Batch
        query_batch = query_vector.reshape(1, 1, -1)  # [1, 1, dim]
        
        # Neuronen als Keys/Values
        kv_batch = neuron_vectors.reshape(1, num_neurons, -1)  # [1, num_neurons, dim]
        
        # Multi-Head Attention
        attention_result = self.multi_head_attention(
            query_vectors=query_batch,
            key_vectors=kv_batch,
            value_vectors=kv_batch
        )
        
        # Residual Connection + Layer Norm
        attended = attention_result.attended_vectors
        normed = self.layer_norm(attended + query_batch)
        
        # Feed-Forward Network
        ff_output = self.feed_forward(normed)
        
        # Residual Connection + Layer Norm
        output = self.layer_norm(ff_output + normed)
        
        # Final context vector
        context_vector = output.squeeze(0).squeeze(0)  # [dim]
        
        return context_vector, attention_result.attention_weights.squeeze(0)
    
    def compute_reasoning_score(
        self,
        query_vector: np.ndarray,
        neuron_vectors: np.ndarray,
        neuron_metadata: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Berechne Reasoning-Scores für Neuronen basierend auf Kontext.
        
        Nutzt echte KI-Logik statt nur Cosine-Similarity.
        
        Args:
            query_vector: Query-Vektor
            neuron_vectors: Neuron-Vektoren
            neuron_metadata: Metadata der Neuronen (tags, etc.)
            
        Returns:
            Reasoning-Scores [num_neurons]
        """
        # Forward pass für kontextuelle Verarbeitung
        context_vector, attention_weights = self.forward_pass(
            neuron_vectors, query_vector
        )
        
        # Berechne Relevanz basierend auf:
        # 1. Attention-Gewichte (wie wichtig ist das Neuron im Kontext)
        # 2. Semantic Similarity zum Context-Vector
        # 3. Metadata-Boost (z.B. für Reasoning-Neuronen)
        
        # 1. Attention-Gewichte (bereits normalisiert)
        attention_scores = attention_weights
        
        # 2. Similarity zum Context-Vector
        context_similarities = self._cosine_similarity(
            neuron_vectors, context_vector.reshape(1, -1)
        ).flatten()
        
        # 3. Metadata-Boost
        metadata_boosts = np.ones(len(neuron_vectors))
        for i, metadata in enumerate(neuron_metadata):
            tags = metadata.get('tags', [])
            # Boost für Reasoning-Neuronen
            if any(tag in ['reasoning', 'logic', 'analysis'] for tag in tags):
                metadata_boosts[i] *= 1.3
            # Boost für Q&A-Neuronen
            if 'qa' in tags or 'conversation' in tags:
                metadata_boosts[i] *= 1.2
        
        # Kombiniere alle Scores
        final_scores = (
            0.4 * attention_scores +
            0.4 * context_similarities +
            0.2 * metadata_boosts
        )
        
        # Normalisiere auf [0, 1]
        final_scores = (final_scores - final_scores.min()) / (final_scores.max() - final_scores.min() + 1e-8)
        
        return final_scores
    
    # === Hilfsfunktionen ===
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation function."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity between vectors."""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.dot(a_norm, b_norm.T)
    
    def _dropout(self, x: np.ndarray, rate: float) -> np.ndarray:
        """Dropout (nur während Training)."""
        if rate == 0.0:
            return x
        mask = np.random.binomial(1, 1 - rate, x.shape)
        return x * mask / (1 - rate)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'embedding_dim': self.embedding_dim,
            'num_attention_heads': self.num_attention_heads,
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,
            'initialized': self.query_weights is not None,
            'query_weights_shape': self.query_weights.shape if self.query_weights is not None else None,
            'ff_weights_shape': self.ff_weights_1.shape if self.ff_weights_1 is not None else None,
        }
