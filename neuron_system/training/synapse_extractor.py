"""
Synapse Extractor - Extract connection patterns from LLM weight matrices.

This extracts the CONNECTIONS between neurons from the model's weights,
not just the neurons themselves. This preserves the model's "wiring diagram".
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class SynapseExtractor:
    """
    Extracts synaptic connections from LLM weight matrices.
    
    Process:
    1. Load model weights
    2. Analyze weight matrices to find strong connections
    3. Map connections to neuron pairs
    4. Store as synapses in Friday's brain
    """
    
    def __init__(self, training_manager):
        """
        Initialize synapse extractor.
        
        Args:
            training_manager: TrainingManager instance
        """
        self.training_manager = training_manager
        self.graph = training_manager.graph
        self.qwen_model = None
        self.neuron_mapping = {}  # Maps Qwen neurons to Friday neurons
        
        # Initialize synapse store
        from neuron_system.storage.synapse_store import SynapseStore
        self.synapse_store = SynapseStore(training_manager.db_manager)
    
    def extract_from_qwen(
        self,
        model_path: str = "models/Qwen_Qwen3-0.6B",
        threshold: float = 0.1,
        max_connections_per_neuron: int = 50
    ) -> Dict[str, Any]:
        """
        Extract synaptic connections from Qwen3 model.
        
        Args:
            model_path: Path to Qwen model
            threshold: Minimum weight strength to create synapse (0.0-1.0)
            max_connections_per_neuron: Limit connections per neuron
            
        Returns:
            Statistics dict
        """
        logger.info("🔗 SYNAPSE EXTRACTION INITIATED")
        logger.info(f"   Threshold: {threshold}")
        logger.info(f"   Max connections per neuron: {max_connections_per_neuron}")
        
        # Load model
        if not self._load_qwen(model_path):
            return {"error": "Failed to load model"}
        
        # Get all neurons
        neurons = list(self.graph.neurons.values())
        logger.info(f"   Found {len(neurons)} neurons to connect")
        
        # Extract connections from different layers
        stats = {
            "total_synapses": 0,
            "layers_processed": 0,
            "average_connections_per_neuron": 0
        }
        
        # Process attention layers
        logger.info("\n⚡ Extracting from Attention Layers...")
        attention_synapses = self._extract_attention_connections(
            neurons, threshold, max_connections_per_neuron
        )
        stats["attention_synapses"] = attention_synapses
        stats["total_synapses"] += attention_synapses
        
        # Process MLP layers
        logger.info("\n🧠 Extracting from MLP Layers...")
        mlp_synapses = self._extract_mlp_connections(
            neurons, threshold, max_connections_per_neuron
        )
        stats["mlp_synapses"] = mlp_synapses
        stats["total_synapses"] += mlp_synapses
        
        # Calculate average
        if len(neurons) > 0:
            stats["average_connections_per_neuron"] = stats["total_synapses"] / len(neurons)
        
        logger.info(f"\n✅ SYNAPSE EXTRACTION COMPLETE")
        logger.info(f"   Total synapses created: {stats['total_synapses']}")
        logger.info(f"   Average per neuron: {stats['average_connections_per_neuron']:.1f}")
        
        return stats
    
    def _load_qwen(self, model_path: str) -> bool:
        """Load Qwen model."""
        try:
            from transformers import AutoModelForCausalLM
            
            possible_paths = [
                "models/Qwen_Qwen3-0.6B/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
                model_path,
                "Qwen/Qwen3-0.6B"
            ]
            
            for path in possible_paths:
                try:
                    self.qwen_model = AutoModelForCausalLM.from_pretrained(
                        path,
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                    logger.info(f"✓ Loaded model from: {path}")
                    return True
                except:
                    continue
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _extract_attention_connections(
        self,
        neurons: List,
        threshold: float,
        max_connections: int
    ) -> int:
        """
        Extract connections from attention weight matrices.
        
        Attention weights show which tokens/concepts attend to each other.
        """
        total_synapses = 0
        
        try:
            # Get attention layers
            for layer_idx, layer in enumerate(self.qwen_model.model.layers):
                if not hasattr(layer, 'self_attn'):
                    continue
                
                # Get query, key, value weight matrices
                q_weight = layer.self_attn.q_proj.weight.data
                k_weight = layer.self_attn.k_proj.weight.data
                v_weight = layer.self_attn.v_proj.weight.data
                
                # Compute attention pattern (simplified)
                # Q * K^T gives attention scores
                # Use more samples to get actual connections
                sample_size = min(len(neurons), q_weight.shape[0], 200)
                
                attention_pattern = torch.matmul(
                    q_weight[:sample_size, :sample_size],
                    k_weight[:sample_size, :sample_size].T
                )
                
                # Normalize
                attention_pattern = torch.softmax(attention_pattern, dim=-1)
                
                # Create synapses for strong connections
                synapses_created = self._create_synapses_from_matrix(
                    attention_pattern,
                    neurons,
                    threshold,
                    max_connections,
                    connection_type="attention"
                )
                
                total_synapses += synapses_created
                
                if (layer_idx + 1) % 5 == 0:
                    logger.info(f"   Layer {layer_idx + 1}: {synapses_created} synapses")
        
        except Exception as e:
            logger.warning(f"Attention extraction failed: {e}")
        
        return total_synapses
    
    def _extract_mlp_connections(
        self,
        neurons: List,
        threshold: float,
        max_connections: int
    ) -> int:
        """
        Extract connections from MLP (feed-forward) weight matrices.
        
        MLP weights show how information flows through the network.
        """
        total_synapses = 0
        
        try:
            # Get MLP layers
            for layer_idx, layer in enumerate(self.qwen_model.model.layers):
                if not hasattr(layer, 'mlp'):
                    continue
                
                # Get MLP weights
                up_weight = layer.mlp.up_proj.weight.data
                down_weight = layer.mlp.down_proj.weight.data
                
                # Combine up and down projections
                # This shows the overall transformation
                sample_size = min(len(neurons), down_weight.shape[0], 200)
                
                # Just use the down projection weights directly (simpler and more meaningful)
                combined = down_weight[:sample_size, :sample_size]
                
                # Normalize
                combined = torch.abs(combined)
                combined = combined / (combined.max() + 1e-8)
                
                # Create synapses
                synapses_created = self._create_synapses_from_matrix(
                    combined,
                    neurons,
                    threshold,
                    max_connections,
                    connection_type="mlp"
                )
                
                total_synapses += synapses_created
                
                if (layer_idx + 1) % 5 == 0:
                    logger.info(f"   Layer {layer_idx + 1}: {synapses_created} synapses")
        
        except Exception as e:
            logger.warning(f"MLP extraction failed: {e}")
        
        return total_synapses
    
    def _create_synapses_from_matrix(
        self,
        weight_matrix: torch.Tensor,
        neurons: List,
        threshold: float,
        max_connections: int,
        connection_type: str
    ) -> int:
        """
        Create synapses from a weight matrix.
        
        Args:
            weight_matrix: PyTorch tensor of weights
            neurons: List of neurons to connect
            threshold: Minimum weight to create connection
            max_connections: Max connections per neuron
            connection_type: Type of connection (attention, mlp, etc.)
            
        Returns:
            Number of synapses created
        """
        synapses_created = 0
        
        try:
            # Convert to numpy
            weights = weight_matrix.cpu().numpy()
            n_rows, n_cols = weights.shape
            
            # Map matrix indices to neurons
            # Use modulo to wrap around if we have fewer neurons than matrix size
            n_neurons = len(neurons)
            
            # Process each row (source neuron)
            for i in range(min(n_rows, n_neurons)):
                source_neuron = neurons[i]
                
                # Get top connections for this neuron
                row = weights[i]
                top_indices = np.argsort(np.abs(row))[-max_connections:]
                
                connections_added = 0
                for j in top_indices:
                    if j >= n_neurons:
                        continue
                    
                    weight = float(row[j])
                    
                    # Only create if above threshold
                    if abs(weight) >= threshold and i != j:
                        target_neuron = neurons[j % n_neurons]
                        
                        # Create synapse using proper API
                        from neuron_system.core.synapse import Synapse, SynapseType
                        from uuid import uuid4
                        
                        synapse = Synapse(
                            id=uuid4(),
                            source_neuron_id=source_neuron.id,
                            target_neuron_id=target_neuron.id,
                            weight=weight,
                            synapse_type=SynapseType.KNOWLEDGE  # Use enum
                        )
                        
                        # Save to database
                        self.synapse_store.create(synapse)
                        
                        synapses_created += 1
                        connections_added += 1
                        
                        if connections_added >= max_connections:
                            break
        
        except Exception as e:
            logger.debug(f"Failed to create synapses: {e}")
        
        return synapses_created
    
    def extract_semantic_connections(
        self,
        similarity_threshold: float = 0.7,
        max_connections: int = 20
    ) -> int:
        """
        Create connections based on semantic similarity of neurons.
        
        This is an alternative/complement to weight-based extraction.
        Neurons with similar embeddings get connected.
        
        Args:
            similarity_threshold: Minimum cosine similarity (0-1)
            max_connections: Max connections per neuron
            
        Returns:
            Number of synapses created
        """
        logger.info("🧬 Creating semantic connections...")
        
        neurons = list(self.graph.neurons.values())
        synapses_created = 0
        
        # Get embeddings
        embeddings = []
        for neuron in neurons:
            if hasattr(neuron, 'vector') and neuron.vector is not None:
                embeddings.append(neuron.vector)
            else:
                embeddings.append(np.zeros(384))  # Default embedding size
        
        embeddings = np.array(embeddings)
        
        # Compute similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # Create connections
        for i, source_neuron in enumerate(neurons):
            # Get most similar neurons
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[-max_connections-1:-1]  # Exclude self
            
            for j in top_indices:
                if similarities[j] >= similarity_threshold and i != j:
                    target_neuron = neurons[j]
                    
                    # Create synapse using proper API
                    from neuron_system.core.synapse import Synapse, SynapseType
                    from uuid import uuid4
                    
                    # Clamp weight to valid range to avoid floating point errors
                    weight = float(similarities[j])
                    weight = max(-1.0, min(1.0, weight))
                    
                    synapse = Synapse(
                        id=uuid4(),
                        source_neuron_id=source_neuron.id,
                        target_neuron_id=target_neuron.id,
                        weight=weight,
                        synapse_type=SynapseType.KNOWLEDGE  # Use enum
                    )
                    
                    self.synapse_store.create(synapse)
                    synapses_created += 1
        
        logger.info(f"✓ Created {synapses_created} semantic connections")
        
        return synapses_created
