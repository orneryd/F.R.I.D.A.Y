"""
Abstract Neuron base class and type registry.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Type
from uuid import UUID
from datetime import datetime
import numpy as np

from neuron_system.core.vector3d import Vector3D


class NeuronType(Enum):
    """Enumeration of neuron types."""
    KNOWLEDGE = "knowledge"
    TOOL = "tool"
    MEMORY = "memory"
    SENSOR = "sensor"
    DECISION = "decision"


class Neuron(ABC):
    """
    Abstract base class for all neuron types.
    
    All neurons must implement process_activation, to_dict, and from_dict methods.
    """
    
    def __init__(self):
        self.id: UUID = None
        self.position: Vector3D = None
        self.vector: np.ndarray = None
        self.neuron_type: NeuronType = None
        self.metadata: Dict[str, Any] = {}
        self.activation_level: float = 0.0
        self.created_at: datetime = None
        self.modified_at: datetime = None
    
    @abstractmethod
    def process_activation(self, activation: float, context: Dict[str, Any]) -> Any:
        """
        Process activation and return result (type-specific behavior).
        
        Args:
            activation: Activation level (0.0 to 1.0)
            context: Additional context for processing
            
        Returns:
            Type-specific result
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize neuron to dictionary.
        
        Returns:
            Dictionary representation
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Neuron':
        """
        Deserialize neuron from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Neuron instance
        """
        pass
    
    def _base_to_dict(self) -> Dict[str, Any]:
        """Helper method to serialize common fields."""
        # Handle vector serialization safely
        vector_data = None
        if self.vector is not None:
            if isinstance(self.vector, np.ndarray):
                vector_data = self.vector.tolist()
            elif isinstance(self.vector, (list, tuple)):
                vector_data = list(self.vector)
            else:
                # Try to convert to numpy array first
                try:
                    vector_data = np.asarray(self.vector).tolist()
                except:
                    vector_data = None
        
        return {
            "id": str(self.id) if self.id else None,
            "type": self.neuron_type.value if self.neuron_type else None,
            "position": self.position.to_dict() if self.position else None,
            "vector": vector_data,
            "metadata": self.metadata,
            "activation_level": self.activation_level,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
        }
    
    def _base_from_dict(self, data: Dict[str, Any]):
        """Helper method to deserialize common fields."""
        if data.get("id"):
            self.id = UUID(data["id"])
        if data.get("position"):
            self.position = Vector3D.from_dict(data["position"])
        if data.get("vector"):
            self.vector = np.array(data["vector"])
        self.metadata = data.get("metadata", {})
        self.activation_level = data.get("activation_level", 0.0)
        if data.get("created_at"):
            self.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("modified_at"):
            self.modified_at = datetime.fromisoformat(data["modified_at"])


class NeuronTypeRegistry:
    """
    Registry for dynamically adding and managing neuron types.
    
    Allows extensibility by registering new neuron types at runtime.
    """
    
    _registry: Dict[str, Type[Neuron]] = {}
    
    @classmethod
    def register(cls, neuron_type: str, neuron_class: Type[Neuron]):
        """
        Register a new neuron type.
        
        Args:
            neuron_type: String identifier for the neuron type
            neuron_class: Class that implements Neuron interface
            
        Raises:
            ValueError: If neuron_class doesn't inherit from Neuron
        """
        if not issubclass(neuron_class, Neuron):
            raise ValueError(f"{neuron_class} must inherit from Neuron")
        cls._registry[neuron_type] = neuron_class
    
    @classmethod
    def create(cls, neuron_type: str, **kwargs) -> Neuron:
        """
        Factory method to create neurons by type.
        
        Args:
            neuron_type: String identifier for the neuron type
            **kwargs: Arguments to pass to neuron constructor
            
        Returns:
            Neuron instance
            
        Raises:
            ValueError: If neuron type is not registered
        """
        if neuron_type not in cls._registry:
            raise ValueError(f"Unknown neuron type: {neuron_type}")
        return cls._registry[neuron_type](**kwargs)
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Neuron:
        """
        Deserialize neuron from dictionary.
        
        Args:
            data: Dictionary representation with 'type' field
            
        Returns:
            Neuron instance
            
        Raises:
            ValueError: If neuron type is not registered
        """
        neuron_type = data.get("type")
        if neuron_type not in cls._registry:
            raise ValueError(f"Unknown neuron type: {neuron_type}")
        return cls._registry[neuron_type].from_dict(data)
    
    @classmethod
    def get_registered_types(cls) -> list:
        """
        Get list of all registered neuron types.
        
        Returns:
            List of registered type names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, neuron_type: str) -> bool:
        """
        Check if a neuron type is registered.
        
        Args:
            neuron_type: String identifier for the neuron type
            
        Returns:
            True if registered, False otherwise
        """
        return neuron_type in cls._registry
