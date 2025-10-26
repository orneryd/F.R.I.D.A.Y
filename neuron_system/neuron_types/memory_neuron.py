"""
Memory Neuron for storing conversation context and history.

Memory neurons enable the AI to remember previous interactions,
maintain conversation context, and provide personalized responses.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID

from neuron_system.core.neuron import Neuron, NeuronType, NeuronTypeRegistry
from neuron_system.core.vector3d import Vector3D


class MemoryNeuron(Neuron):
    """
    Specialized neuron for storing conversation memory and context.
    
    Memory neurons store:
    - Conversation history
    - User preferences
    - Context from previous interactions
    - Temporal information
    
    Attributes:
        memory_type: Type of memory (short-term, long-term, episodic, semantic)
        timestamp: When the memory was created
        context: Additional context information
        importance: Importance score (0.0 to 1.0)
        access_count: How many times this memory has been accessed
        last_accessed: When the memory was last accessed
    """
    
    def __init__(
        self,
        memory_type: str = "short-term",
        timestamp: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        source_data: str = "",
        **kwargs
    ):
        """
        Initialize a memory neuron.
        
        Args:
            memory_type: Type of memory (short-term, long-term, episodic, semantic)
            timestamp: When the memory was created
            context: Additional context information
            importance: Importance score (0.0 to 1.0)
            source_data: Memory content
            **kwargs: Additional arguments passed to Neuron
        """
        super().__init__(**kwargs)
        self.neuron_type = NeuronType.MEMORY  # Set neuron type to MEMORY
        
        # Ensure ID is set
        from uuid import uuid4
        if not self.id:
            self.id = uuid4()
        if not self.created_at:
            self.created_at = datetime.now()
        if not self.modified_at:
            self.modified_at = datetime.now()
        
        self.memory_type = memory_type
        self.timestamp = timestamp or datetime.now()
        self.context = context or {}
        self.importance = max(0.0, min(1.0, importance))  # Clamp to [0, 1]
        self.access_count = 0
        self.last_accessed = None
        self.source_data = source_data  # Store memory content
    
    def process_activation(self, activation: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process activation and return memory content.
        
        Args:
            activation: Activation level (0.0 to 1.0)
            context: Additional context
            
        Returns:
            Dictionary with memory information
        """
        self.activation_level = activation
        self.access()  # Record access
        
        return {
            "type": "memory",
            "neuron_id": str(self.id),
            "content": self.source_data,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "activation": activation,
            "access_count": self.access_count,
            "timestamp": self.timestamp.isoformat()
        }
    
    def access(self):
        """Record that this memory was accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def decay(self, decay_rate: float = 0.1):
        """
        Apply memory decay (reduce importance over time).
        
        Args:
            decay_rate: Rate of decay (0.0 to 1.0)
        """
        self.importance *= (1.0 - decay_rate)
        self.importance = max(0.0, self.importance)
    
    def reinforce(self, reinforcement: float = 0.1):
        """
        Reinforce memory (increase importance).
        
        Args:
            reinforcement: Amount to reinforce (0.0 to 1.0)
        """
        self.importance = min(1.0, self.importance + reinforcement)
    
    def is_expired(self, max_age_seconds: Optional[int] = None) -> bool:
        """
        Check if memory has expired based on age.
        
        Args:
            max_age_seconds: Maximum age in seconds (None = never expires)
            
        Returns:
            True if expired, False otherwise
        """
        if max_age_seconds is None:
            return False
        
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > max_age_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert memory neuron to dictionary.
        
        Returns:
            Dictionary representation
        """
        base_dict = self._base_to_dict()  # Use _base_to_dict instead of super().to_dict()
        base_dict.update({
            'neuron_class': 'MemoryNeuron',  # Mark as MemoryNeuron for deserialization
            'memory_type': self.memory_type,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'importance': self.importance,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'source_data': self.source_data  # Include source_data
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNeuron':
        """
        Create memory neuron from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            MemoryNeuron instance
        """
        # Make a copy to avoid modifying original
        data = data.copy()
        
        # Extract memory-specific fields
        memory_type = data.pop('memory_type', 'short-term')
        timestamp_str = data.pop('timestamp', None)
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else None
        context = data.pop('context', {})
        importance = data.pop('importance', 0.5)
        access_count = data.pop('access_count', 0)
        last_accessed_str = data.pop('last_accessed', None)
        source_data = data.pop('source_data', '')
        
        # Create neuron
        neuron = cls(
            memory_type=memory_type,
            timestamp=timestamp,
            context=context,
            importance=importance,
            source_data=source_data
        )
        
        # Restore base neuron fields
        neuron._base_from_dict(data)
        
        # Restore memory-specific fields
        neuron.access_count = access_count
        if last_accessed_str:
            neuron.last_accessed = datetime.fromisoformat(last_accessed_str)
        
        return neuron
    
    def __repr__(self) -> str:
        """String representation of memory neuron."""
        return (
            f"MemoryNeuron(id={self.id}, type={self.memory_type}, "
            f"importance={self.importance:.2f}, access_count={self.access_count})"
        )


class MemoryManager:
    """
    Manages memory neurons for conversation context.
    
    Handles:
    - Creating and storing memories
    - Retrieving relevant memories
    - Memory decay and consolidation
    - Context management
    """
    
    def __init__(self, graph, compression_engine):
        """
        Initialize memory manager.
        
        Args:
            graph: NeuronGraph instance
            compression_engine: CompressionEngine for encoding memories
        """
        self.graph = graph
        self.compression_engine = compression_engine
        self.short_term_max_age = 3600  # 1 hour
        self.long_term_threshold = 0.7  # Importance threshold for long-term storage
    
    def create_memory(
        self,
        content: str,
        memory_type: str = "short-term",
        context: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> UUID:
        """
        Create a new memory neuron.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            context: Additional context
            importance: Importance score
            
        Returns:
            ID of created memory neuron
        """
        # Compress content to vector
        vector, metadata = self.compression_engine.compress(content)
        
        # Generate position
        position = self._generate_memory_position(memory_type)
        
        # Create memory neuron
        memory = MemoryNeuron(
            memory_type=memory_type,
            timestamp=datetime.now(),
            context=context,
            importance=importance
        )
        
        memory.position = position
        memory.vector = vector
        memory.source_data = content
        
        # Add to graph
        self.graph.add_neuron(memory)
        
        return memory.id
    
    def retrieve_memories(
        self,
        query: str,
        memory_type: Optional[str] = None,
        top_k: int = 5,
        min_importance: float = 0.3
    ) -> List[MemoryNeuron]:
        """
        Retrieve relevant memories based on query.
        
        Args:
            query: Query text
            memory_type: Filter by memory type (None = all types)
            top_k: Number of memories to retrieve
            min_importance: Minimum importance threshold
            
        Returns:
            List of relevant memory neurons
        """
        # Get all memory neurons
        memories = [
            n for n in self.graph.neurons.values()
            if isinstance(n, MemoryNeuron)
            and n.importance >= min_importance
            and (memory_type is None or n.memory_type == memory_type)
            and not n.is_expired(self.short_term_max_age if n.memory_type == "short-term" else None)
        ]
        
        if not memories:
            return []
        
        # Compress query
        query_vector, _ = self.compression_engine.compress(query)
        
        # Calculate similarity and sort
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        scored_memories = []
        for memory in memories:
            if memory.vector is not None:
                similarity = cosine_similarity(
                    [query_vector],
                    [memory.vector]
                )[0][0]
                
                # Combine similarity with importance and recency
                age_factor = 1.0
                if memory.last_accessed:
                    age_seconds = (datetime.now() - memory.last_accessed).total_seconds()
                    age_factor = max(0.5, 1.0 - (age_seconds / 86400))  # Decay over 24 hours
                
                score = similarity * memory.importance * age_factor
                scored_memories.append((memory, score))
        
        # Sort by score and return top-k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        result = [m for m, _ in scored_memories[:top_k]]
        
        # Mark as accessed
        for memory in result:
            memory.access()
        
        return result
    
    def consolidate_memories(self):
        """
        Consolidate short-term memories into long-term storage.
        
        Promotes important short-term memories to long-term
        and removes unimportant expired memories.
        """
        memories = [
            n for n in self.graph.neurons.values()
            if isinstance(n, MemoryNeuron)
        ]
        
        for memory in memories:
            if memory.memory_type == "short-term":
                # Check if should be promoted to long-term
                if memory.importance >= self.long_term_threshold:
                    memory.memory_type = "long-term"
                    memory.reinforce(0.1)
                
                # Remove expired unimportant memories
                elif memory.is_expired(self.short_term_max_age) and memory.importance < 0.3:
                    self.graph.remove_neuron(memory.id)
            
            # Apply decay to all memories
            memory.decay(0.05)
    
    def _generate_memory_position(self, memory_type: str) -> Vector3D:
        """
        Generate position for memory neuron based on type.
        
        Args:
            memory_type: Type of memory
            
        Returns:
            Position vector
        """
        import numpy as np
        
        # Different regions for different memory types
        if memory_type == "short-term":
            base = Vector3D(100, 100, 100)
        elif memory_type == "long-term":
            base = Vector3D(-100, -100, -100)
        elif memory_type == "episodic":
            base = Vector3D(100, -100, 0)
        else:  # semantic
            base = Vector3D(-100, 100, 0)
        
        # Add some randomness
        offset = Vector3D(
            np.random.uniform(-20, 20),
            np.random.uniform(-20, 20),
            np.random.uniform(-20, 20)
        )
        
        return base + offset
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        memories = [
            n for n in self.graph.neurons.values()
            if isinstance(n, MemoryNeuron)
        ]
        
        if not memories:
            return {
                'total_memories': 0,
                'by_type': {},
                'average_importance': 0.0,
                'total_accesses': 0
            }
        
        by_type = {}
        for memory in memories:
            by_type[memory.memory_type] = by_type.get(memory.memory_type, 0) + 1
        
        return {
            'total_memories': len(memories),
            'by_type': by_type,
            'average_importance': sum(m.importance for m in memories) / len(memories),
            'total_accesses': sum(m.access_count for m in memories)
        }


# Register MemoryNeuron in the type registry
NeuronTypeRegistry.register("memory", MemoryNeuron)
