# Custom Neuron Type Extension Guide

This guide demonstrates how to extend the 3D Synaptic Neuron System with custom neuron types. We use `MemoryNeuron` as a complete example.

## Overview

The neuron system is designed to be extensible. You can create new neuron types by:

1. Creating a new neuron class that inherits from `Neuron`
2. Implementing required abstract methods
3. Registering the type in the `NeuronTypeRegistry`
4. Adding API support (optional)
5. Updating serialization/deserialization

## Step 1: Create Your Neuron Class

Create a new file in `neuron_system/neuron_types/` for your custom neuron type.

### Example: MemoryNeuron

```python
# neuron_system/neuron_types/memory_neuron.py

from typing import Any, Dict, List
from uuid import uuid4
from datetime import datetime
import numpy as np

from neuron_system.core.neuron import Neuron, NeuronType, NeuronTypeRegistry
from neuron_system.core.vector3d import Vector3D


class MemoryNeuron(Neuron):
    """
    Stores temporal sequences and episodic memories.
    
    Custom neuron type for storing time-ordered sequences of events.
    """
    
    def __init__(self,
                 sequence_data: List[Dict[str, Any]] = None,
                 temporal_index: int = 0,
                 retention_strength: float = 1.0,
                 memory_type: str = "episodic",
                 max_sequence_length: int = 100,
                 **kwargs):
        """Initialize with custom fields."""
        super().__init__()
        self.neuron_type = NeuronType.MEMORY
        
        # Custom fields for this neuron type
        self.sequence_data = sequence_data or []
        self.temporal_index = temporal_index
        self.retention_strength = retention_strength
        self.memory_type = memory_type
        self.max_sequence_length = max_sequence_length
        
        # Set defaults
        if not self.id:
            self.id = uuid4()
        if not self.created_at:
            self.created_at = datetime.now()
        if not self.modified_at:
            self.modified_at = datetime.now()
```

## Step 2: Implement Required Methods

Every neuron type must implement three abstract methods:

### 2.1 process_activation()

This method defines how the neuron behaves when activated during queries.

```python
def process_activation(self, activation: float, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Define custom activation behavior.
    
    Args:
        activation: Activation level (0.0 to 1.0)
        context: Additional context for processing
        
    Returns:
        Dictionary with type-specific results
    """
    self.activation_level = activation
    
    # Custom logic: retrieve memories based on activation
    effective_activation = activation * self.retention_strength
    
    retrieval_mode = context.get('retrieval_mode', 'recent')
    max_items = context.get('max_items', 10)
    
    # Your custom retrieval logic here
    retrieved_sequence = self._retrieve_memories(retrieval_mode, max_items)
    
    return {
        "type": "memory",
        "neuron_id": str(self.id),
        "sequence": retrieved_sequence,
        "activation": activation,
        "retention_strength": self.retention_strength,
    }
```

### 2.2 to_dict()

Serialize your neuron to a dictionary for storage.

```python
def to_dict(self) -> Dict[str, Any]:
    """Serialize neuron to dictionary."""
    # Start with base fields
    base_dict = self._base_to_dict()
    
    # Add custom fields
    base_dict.update({
        "sequence_data": self.sequence_data,
        "temporal_index": self.temporal_index,
        "retention_strength": self.retention_strength,
        "memory_type": self.memory_type,
        "max_sequence_length": self.max_sequence_length,
    })
    
    return base_dict
```

### 2.3 from_dict()

Deserialize your neuron from a dictionary.

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNeuron':
    """Deserialize neuron from dictionary."""
    # Create instance with custom fields
    neuron = cls(
        sequence_data=data.get("sequence_data", []),
        temporal_index=data.get("temporal_index", 0),
        retention_strength=data.get("retention_strength", 1.0),
        memory_type=data.get("memory_type", "episodic"),
        max_sequence_length=data.get("max_sequence_length", 100),
    )
    
    # Load base fields
    neuron._base_from_dict(data)
    
    return neuron
```

## Step 3: Register Your Neuron Type

At the end of your neuron file, register it with the type registry:

```python
# Register MemoryNeuron in the type registry
NeuronTypeRegistry.register("memory", MemoryNeuron)
```

This allows the system to:
- Create neurons of your type using `NeuronTypeRegistry.create("memory", ...)`
- Deserialize neurons from storage
- Validate neuron types

## Step 4: Update Module Exports

Add your neuron to `neuron_system/neuron_types/__init__.py`:

```python
from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
from neuron_system.neuron_types.tool_neuron import ToolNeuron
from neuron_system.neuron_types.memory_neuron import MemoryNeuron

__all__ = [
    "KnowledgeNeuron",
    "ToolNeuron",
    "MemoryNeuron",
]
```

## Step 5: Add API Support (Optional)

To create your custom neurons via the REST API:

### 5.1 Update API Models

Add fields to `neuron_system/api/models.py`:

```python
class NeuronCreateRequest(BaseModel):
    neuron_type: str
    # ... existing fields ...
    
    # Memory neuron specific
    sequence_data: Optional[List[Dict[str, Any]]] = None
    memory_type: Optional[str] = "episodic"
    retention_strength: Optional[float] = 1.0
    max_sequence_length: Optional[int] = 100
```

```python
class NeuronResponse(BaseModel):
    # ... existing fields ...
    
    # Memory neuron specific
    sequence_data: Optional[List[Dict[str, Any]]] = None
    memory_type: Optional[str] = None
    retention_strength: Optional[float] = None
    temporal_index: Optional[int] = None
```

### 5.2 Update Neurons Route

Add handling in `neuron_system/api/routes/neurons.py`:

```python
from neuron_system.neuron_types.memory_neuron import MemoryNeuron

# In neuron_to_response():
elif isinstance(neuron, MemoryNeuron):
    response_data["sequence_data"] = neuron.sequence_data
    response_data["memory_type"] = neuron.memory_type
    response_data["retention_strength"] = neuron.retention_strength
    response_data["temporal_index"] = neuron.temporal_index

# In create_neuron():
elif request.neuron_type == "memory":
    neuron = state.graph.add_neuron(
        neuron_type="memory",
        position=position,
        sequence_data=request.sequence_data or [],
        memory_type=request.memory_type or "episodic",
        retention_strength=request.retention_strength or 1.0,
        max_sequence_length=request.max_sequence_length or 100,
        metadata=request.metadata
    )
```

## Step 6: Add Custom Methods (Optional)

Add domain-specific methods to your neuron class:

```python
class MemoryNeuron(Neuron):
    # ... existing code ...
    
    def add_memory(self, memory_data: Dict[str, Any]) -> None:
        """Add a new memory to the sequence."""
        if 'timestamp' not in memory_data:
            memory_data['timestamp'] = datetime.now().isoformat()
        
        self.sequence_data.append(memory_data)
        self.temporal_index = len(self.sequence_data) - 1
        
        # Enforce max length
        if len(self.sequence_data) > self.max_sequence_length:
            self.sequence_data = self.sequence_data[-self.max_sequence_length:]
        
        self.modified_at = datetime.now()
    
    def decay_retention(self, decay_rate: float = 0.01) -> None:
        """Simulate memory fading over time."""
        self.retention_strength = max(0.0, self.retention_strength - decay_rate)
        self.modified_at = datetime.now()
```

## Usage Examples

### Creating a Memory Neuron Programmatically

```python
from neuron_system.neuron_types.memory_neuron import MemoryNeuron
from neuron_system.core.vector3d import Vector3D

# Create a memory neuron
memory_neuron = MemoryNeuron(
    sequence_data=[
        {"event": "user_login", "timestamp": "2025-01-01T10:00:00"},
        {"event": "page_view", "page": "/dashboard", "timestamp": "2025-01-01T10:01:00"},
    ],
    memory_type="episodic",
    retention_strength=1.0
)

# Add to graph
graph.add_neuron_instance(memory_neuron)

# Add more memories
memory_neuron.add_memory({"event": "button_click", "button": "submit"})
```

### Creating via API

```bash
curl -X POST http://localhost:8000/api/neurons \
  -H "Content-Type: application/json" \
  -d '{
    "neuron_type": "memory",
    "memory_type": "episodic",
    "sequence_data": [
      {"event": "user_login", "timestamp": "2025-01-01T10:00:00"}
    ],
    "retention_strength": 1.0,
    "max_sequence_length": 100
  }'
```

### Querying Memory Neurons

```python
# Query with custom context
result = query_engine.query(
    "recent user activity",
    context={
        "retrieval_mode": "recent",
        "max_items": 5
    }
)

# Process memory neuron results
for activated in result:
    if activated.neuron.neuron_type == NeuronType.MEMORY:
        memories = activated.result["sequence"]
        print(f"Retrieved {len(memories)} memories")
```

## Best Practices

### 1. Use Meaningful NeuronType Enum

Add your type to the `NeuronType` enum in `neuron_system/core/neuron.py`:

```python
class NeuronType(Enum):
    KNOWLEDGE = "knowledge"
    TOOL = "tool"
    MEMORY = "memory"
    YOUR_TYPE = "your_type"  # Add here
```

### 2. Validate Custom Fields

Add validation in your `__init__` method:

```python
def __init__(self, retention_strength: float = 1.0, **kwargs):
    super().__init__()
    
    # Validate custom fields
    if not 0.0 <= retention_strength <= 1.0:
        raise ValueError("retention_strength must be between 0.0 and 1.0")
    
    self.retention_strength = retention_strength
```

### 3. Document Your Neuron Type

Add comprehensive docstrings:

```python
class MemoryNeuron(Neuron):
    """
    Stores temporal sequences and episodic memories.
    
    Use Cases:
    - Conversation history tracking
    - Event sequence recording
    - Temporal pattern analysis
    
    Custom Fields:
    - sequence_data: List of temporal events
    - retention_strength: Memory retention (0.0 to 1.0)
    - memory_type: Type of memory (episodic, semantic, procedural)
    
    Requirements: 13.1, 13.2, 13.3, 13.4, 13.5
    """
```

### 4. Handle Edge Cases

```python
def process_activation(self, activation: float, context: Dict[str, Any]) -> Dict[str, Any]:
    # Handle empty sequences
    if not self.sequence_data:
        return {
            "type": "memory",
            "neuron_id": str(self.id),
            "sequence": [],
            "message": "No memories stored"
        }
    
    # Handle invalid context
    retrieval_mode = context.get('retrieval_mode', 'recent')
    if retrieval_mode not in ['full', 'recent', 'relevant']:
        retrieval_mode = 'recent'
    
    # Your logic here
```

### 5. Test Your Neuron Type

Create tests in `tests/test_memory_neuron.py`:

```python
def test_memory_neuron_creation():
    neuron = MemoryNeuron(
        sequence_data=[{"event": "test"}],
        memory_type="episodic"
    )
    assert neuron.neuron_type == NeuronType.MEMORY
    assert len(neuron.sequence_data) == 1

def test_memory_neuron_serialization():
    neuron = MemoryNeuron(sequence_data=[{"event": "test"}])
    data = neuron.to_dict()
    restored = MemoryNeuron.from_dict(data)
    assert restored.sequence_data == neuron.sequence_data
```

## Advanced Features

### Custom Activation Propagation

Override how activation propagates through your neurons:

```python
def propagate_to_neighbors(self, graph, activation: float) -> List[Tuple[UUID, float]]:
    """Custom propagation logic for memory neurons."""
    propagated = []
    
    for synapse in graph.get_outgoing_synapses(self.id):
        # Apply temporal decay to propagation
        time_factor = self._calculate_temporal_decay()
        new_activation = activation * synapse.weight * time_factor
        
        propagated.append((synapse.target_neuron_id, new_activation))
    
    return propagated
```

### Integration with Other Neuron Types

Create synapses between your custom type and existing types:

```python
# Connect memory neuron to knowledge neurons
memory_neuron = MemoryNeuron(...)
knowledge_neuron = KnowledgeNeuron(...)

# Create synapse
synapse = Synapse(
    source_neuron_id=memory_neuron.id,
    target_neuron_id=knowledge_neuron.id,
    weight=0.8,
    synapse_type=SynapseType.KNOWLEDGE
)

graph.add_synapse(synapse)
```

## Troubleshooting

### Common Issues

1. **Type not registered**: Ensure you call `NeuronTypeRegistry.register()` at module load time
2. **Serialization errors**: Make sure all custom fields are JSON-serializable
3. **Missing in API**: Update both models and routes to support your type
4. **Import errors**: Check circular dependencies and import order

### Debugging

```python
# Check if type is registered
print(NeuronTypeRegistry.get_registered_types())
# Output: ['knowledge', 'tool', 'memory']

# Verify registration
assert NeuronTypeRegistry.is_registered("memory")

# Test factory creation
neuron = NeuronTypeRegistry.create("memory", sequence_data=[])
assert isinstance(neuron, MemoryNeuron)
```

## Summary

Creating custom neuron types involves:

1. ✅ Create class inheriting from `Neuron`
2. ✅ Implement `process_activation()`, `to_dict()`, `from_dict()`
3. ✅ Register with `NeuronTypeRegistry.register()`
4. ✅ Update module exports
5. ✅ Add API support (optional)
6. ✅ Add custom methods as needed
7. ✅ Write tests

The `MemoryNeuron` implementation in `neuron_system/neuron_types/memory_neuron.py` serves as a complete reference example for creating your own custom neuron types.
