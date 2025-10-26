"""
KnowledgeNeuron implementation for storing compressed knowledge.
"""

from typing import Any, Dict, List
from uuid import UUID, uuid4
from datetime import datetime
import numpy as np

from neuron_system.core.neuron import Neuron, NeuronType, NeuronTypeRegistry
from neuron_system.core.vector3d import Vector3D


class KnowledgeNeuron(Neuron):
    """
    Stores compressed knowledge/information as a neuron.
    
    Knowledge neurons represent semantic information embedded in vector space.
    """
    
    def __init__(self, 
                 source_data: str = "",
                 compression_ratio: float = 0.0,
                 semantic_tags: List[str] = None,
                 position: Vector3D = None,
                 vector: Any = None,
                 importance: float = 0.5,
                 **kwargs):
        """
        Initialize knowledge neuron.
        
        Args:
            source_data: Original compressed data
            compression_ratio: Ratio of compression achieved
            semantic_tags: List of semantic tags for categorization
            position: 3D position in space
            vector: Embedding vector
            importance: Importance/quality score (0.0 to 1.0, default 0.5)
            **kwargs: Additional arguments
        """
        super().__init__()
        self.neuron_type = NeuronType.KNOWLEDGE
        self.source_data = source_data
        self.compression_ratio = compression_ratio
        self.semantic_tags = semantic_tags or []
        self.importance = importance  # For self-training quality tracking
        
        # Set position and vector if provided
        if position is not None:
            self.position = position
        if vector is not None:
            self.vector = vector
        
        # Set defaults
        if not self.id:
            self.id = uuid4()
        if not self.created_at:
            self.created_at = datetime.now()
        if not self.modified_at:
            self.modified_at = datetime.now()
    
    def process_activation(self, activation: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return knowledge content when activated.
        
        Args:
            activation: Activation level (0.0 to 1.0)
            context: Additional context for processing
            
        Returns:
            Dictionary with knowledge content and metadata
        """
        self.activation_level = activation
        
        return {
            "type": "knowledge",
            "neuron_id": str(self.id),
            "content": self.source_data,
            "tags": self.semantic_tags,
            "activation": activation,
            "compression_ratio": self.compression_ratio,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize knowledge neuron to dictionary.
        
        Returns:
            Dictionary representation
        """
        base_dict = self._base_to_dict()
        base_dict.update({
            "source_data": self.source_data,
            "compression_ratio": self.compression_ratio,
            "semantic_tags": self.semantic_tags,
            "importance": self.importance,
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNeuron':
        """
        Deserialize knowledge neuron from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            KnowledgeNeuron instance
        """
        neuron = cls(
            source_data=data.get("source_data", ""),
            compression_ratio=data.get("compression_ratio", 0.0),
            semantic_tags=data.get("semantic_tags", []),
            importance=data.get("importance", 0.5),
        )
        neuron._base_from_dict(data)
        return neuron


# Register KnowledgeNeuron in the type registry
NeuronTypeRegistry.register("knowledge", KnowledgeNeuron)
