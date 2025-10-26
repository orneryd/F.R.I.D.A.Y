"""
Synapse implementation for neuron connections.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict
from uuid import UUID, uuid4
from datetime import datetime


class SynapseType(Enum):
    """Enumeration of synapse types."""
    KNOWLEDGE = "knowledge"
    TOOL_INPUT = "tool_input"
    TOOL_OUTPUT = "tool_output"


@dataclass
class Synapse:
    """
    Represents a directed connection between two neurons.
    
    Synapses have weights that determine connection strength and can be
    strengthened or weakened through usage and training.
    """
    
    id: UUID = field(default_factory=uuid4)
    source_neuron_id: UUID = None
    target_neuron_id: UUID = None
    weight: float = 0.5
    usage_count: int = 0
    last_traversed: datetime = None
    synapse_type: SynapseType = SynapseType.KNOWLEDGE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate synapse parameters after initialization."""
        self._validate_weight()
    
    def _validate_weight(self):
        """Ensure weight is within valid bounds [-1.0, 1.0]."""
        if self.weight < -1.0 or self.weight > 1.0:
            raise ValueError(f"Synapse weight must be between -1.0 and 1.0, got {self.weight}")
    
    def strengthen(self, delta: float = 0.01):
        """
        Increase synapse weight (Hebbian learning).
        
        Args:
            delta: Amount to increase weight
        """
        self.weight = min(1.0, self.weight + delta)
        self.modified_at = datetime.now()
    
    def weaken(self, delta: float = 0.001):
        """
        Decrease synapse weight (decay).
        
        Args:
            delta: Amount to decrease weight
        """
        self.weight = max(-1.0, self.weight - delta)
        self.modified_at = datetime.now()
    
    def traverse(self):
        """
        Mark synapse as traversed during activation propagation.
        
        Increments usage counter and updates last traversed timestamp.
        """
        self.usage_count += 1
        self.last_traversed = datetime.now()
    
    def is_weak(self, threshold: float = 0.1) -> bool:
        """
        Check if synapse is weak (below threshold).
        
        Args:
            threshold: Weight threshold for weakness
            
        Returns:
            True if weight is below threshold
        """
        return abs(self.weight) < threshold
    
    def should_prune(self) -> bool:
        """
        Check if synapse should be deleted (weight near zero).
        
        Returns:
            True if weight is effectively zero
        """
        return abs(self.weight) < 0.01
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize synapse to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": str(self.id),
            "source_neuron_id": str(self.source_neuron_id) if self.source_neuron_id else None,
            "target_neuron_id": str(self.target_neuron_id) if self.target_neuron_id else None,
            "weight": self.weight,
            "usage_count": self.usage_count,
            "last_traversed": self.last_traversed.isoformat() if self.last_traversed else None,
            "synapse_type": self.synapse_type.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Synapse':
        """
        Deserialize synapse from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Synapse instance
        """
        synapse = cls(
            id=UUID(data["id"]) if data.get("id") else uuid4(),
            source_neuron_id=UUID(data["source_neuron_id"]) if data.get("source_neuron_id") else None,
            target_neuron_id=UUID(data["target_neuron_id"]) if data.get("target_neuron_id") else None,
            weight=data.get("weight", 0.5),
            usage_count=data.get("usage_count", 0),
            synapse_type=SynapseType(data.get("synapse_type", "knowledge")),
            metadata=data.get("metadata", {}),
        )
        
        if data.get("last_traversed"):
            synapse.last_traversed = datetime.fromisoformat(data["last_traversed"])
        if data.get("created_at"):
            synapse.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("modified_at"):
            synapse.modified_at = datetime.fromisoformat(data["modified_at"])
        
        return synapse
