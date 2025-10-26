# 3D Synaptic Neuron System

A novel knowledge representation system that models knowledge as a 3D network of neurons and synapses with native tool integration.

## Project Structure

```
neuron_system/
├── __init__.py                    # Main package exports
│
├── core/                          # Core data structures
│   ├── __init__.py
│   ├── vector3d.py               # Vector3D class for 3D positioning
│   ├── neuron.py                 # Abstract Neuron base class + Registry
│   ├── synapse.py                # Synapse class for connections
│   └── graph.py                  # NeuronGraph main class
│
├── neuron_types/                  # Specialized neuron implementations
│   ├── __init__.py
│   ├── knowledge_neuron.py       # KnowledgeNeuron for storing information
│   ├── tool_neuron.py            # ToolNeuron for executable functions
│   └── memory_neuron.py          # MemoryNeuron for temporal sequences (example)
│
├── engines/                       # Processing engines (future tasks)
│   └── __init__.py
│
├── storage/                       # Persistence layer (future tasks)
│   └── __init__.py
│
├── spatial/                       # Spatial indexing (future tasks)
│   └── __init__.py
│
├── tools/                         # Tool system (future tasks)
│   └── __init__.py
│
├── api/                           # REST API (future tasks)
│   └── __init__.py
│
├── sdk/                           # Python SDK (future tasks)
│   └── __init__.py
│
├── utils/                         # Shared utilities
│   ├── __init__.py
│   ├── pooling.py                # Object pooling for memory efficiency
│   └── uuid_pool.py              # UUID pre-allocation
│
└── config/                        # Configuration
    ├── __init__.py
    └── settings.py               # Application settings
```

## Module Dependencies

### Dependency Hierarchy

```
Level 0 (No dependencies):
  - config/
  - utils/

Level 1 (Depends on Level 0):
  - core/vector3d.py

Level 2 (Depends on Level 0-1):
  - core/neuron.py
  - core/synapse.py

Level 3 (Depends on Level 0-2):
  - core/graph.py
  - neuron_types/

Level 4+ (Future tasks):
  - engines/
  - storage/
  - spatial/
  - tools/
  - api/
  - sdk/
```

### Module Responsibilities

| Module | Responsibility | Dependencies |
|--------|---------------|--------------|
| **core/** | Base data structures (Neuron, Synapse, Graph, Vector3D) | None |
| **neuron_types/** | Concrete neuron implementations (Knowledge, Tool) | core/ |
| **engines/** | Processing logic (Compression, Query, Training) | core/, neuron_types/ |
| **storage/** | Persistence and serialization | core/, neuron_types/ |
| **spatial/** | 3D indexing and positioning | core/ |
| **tools/** | Tool execution and clustering | core/, neuron_types/ |
| **api/** | REST API endpoints | engines/, storage/, spatial/ |
| **sdk/** | Python client library | None (HTTP only) |
| **utils/** | Shared utilities (pooling, UUID) | None |
| **config/** | Configuration management | None |

## Core Components

### Vector3D

3D vector implementation for neuron positioning in space.

```python
from neuron_system.core import Vector3D

# Create a 3D position
pos = Vector3D(x=10.0, y=20.0, z=30.0)

# Calculate distance
distance = pos.distance(Vector3D(0, 0, 0))

# Vector operations
normalized = pos.normalize()
magnitude = pos.magnitude()
```

### Neuron

Abstract base class for all neuron types. Supports extensibility through the type registry.

```python
from neuron_system.core import Neuron, NeuronType, NeuronTypeRegistry

# Check registered types
types = NeuronTypeRegistry.get_registered_types()
# ['knowledge', 'tool', 'memory']

# Create neuron by type
neuron = NeuronTypeRegistry.create('knowledge', source_data="Example data")

# Serialize/deserialize
data = neuron.to_dict()
restored = NeuronTypeRegistry.deserialize(data)
```

### Synapse

Directed connection between neurons with weight tracking.

```python
from neuron_system.core import Synapse, SynapseType

# Create synapse
synapse = Synapse(
    source_neuron_id=source_id,
    target_neuron_id=target_id,
    weight=0.8,
    synapse_type=SynapseType.KNOWLEDGE
)

# Modify weight
synapse.strengthen(delta=0.01)
synapse.weaken(delta=0.001)

# Track usage
synapse.traverse()
```

### NeuronGraph

Main graph structure managing neurons and synapses.

```python
from neuron_system.core import NeuronGraph
from neuron_system.neuron_types import KnowledgeNeuron

# Create graph
graph = NeuronGraph()

# Add neurons
neuron = KnowledgeNeuron(source_data="Example")
graph.add_neuron(neuron)

# Add synapses
synapse = Synapse(source_neuron_id=id1, target_neuron_id=id2)
graph.add_synapse(synapse)

# Query connections
neighbors = graph.get_neighbors(neuron.id)
outgoing = graph.get_outgoing_synapses(neuron.id)
```

### KnowledgeNeuron

Stores compressed knowledge as embedded vectors.

```python
from neuron_system.neuron_types import KnowledgeNeuron

neuron = KnowledgeNeuron(
    source_data="Machine learning is a subset of AI",
    compression_ratio=0.95,
    semantic_tags=["AI", "ML"]
)

# Process activation
result = neuron.process_activation(activation=0.8, context={})
# Returns: {"type": "knowledge", "content": "...", "tags": [...]}
```

### ToolNeuron

Represents executable functions in the network.

```python
from neuron_system.neuron_types import ToolNeuron

neuron = ToolNeuron(
    function_signature="calculate(x: int, y: int) -> int",
    executable_code="return x + y",
    input_schema={"x": "int", "y": "int"},
    activation_threshold=0.5
)

# Process activation (executes if threshold met)
result = neuron.process_activation(
    activation=0.7,
    context={"inputs": {"x": 5, "y": 3}}
)
```

## Utilities

### ObjectPool

Generic object pooling for memory efficiency.

```python
from neuron_system.utils import ObjectPool

# Create pool with factory function
pool = ObjectPool(factory=lambda: MyObject(), max_size=1000)

# Acquire and release objects
obj = pool.acquire()
# ... use object ...
pool.release(obj)

# Get statistics
stats = pool.get_stats()
```

### UUIDPool

Pre-allocated UUID pool for instant ID assignment.

```python
from neuron_system.utils import UUIDPool

# Create pool
uuid_pool = UUIDPool(initial_size=10000)

# Get UUID instantly (< 0.01ms)
neuron_id = uuid_pool.acquire()

# Check pool status
stats = uuid_pool.get_stats()
```

## Configuration

Centralized configuration management.

```python
from neuron_system.config import get_settings, update_settings

# Get settings
settings = get_settings()
print(settings.vector_dimensions)  # 384
print(settings.default_top_k)      # 10

# Update settings
update_settings(
    default_top_k=20,
    query_timeout_seconds=10.0
)
```

## Design Principles

1. **Single Responsibility**: Each file/class has exactly one purpose
2. **Clear Dependencies**: Modules import only from "lower" levels (no circular imports)
3. **Extensibility**: New neuron types can be registered at runtime
4. **Type Safety**: Abstract base classes enforce interface contracts
5. **Performance**: Object pooling and UUID pre-allocation for high-frequency operations

## Extensibility Example

The system includes a complete example of custom neuron type extension with `MemoryNeuron`.

### MemoryNeuron

Stores temporal sequences and episodic memories.

```python
from neuron_system.neuron_types import MemoryNeuron

# Create a memory neuron
neuron = MemoryNeuron(
    sequence_data=[
        {"event": "user_login", "timestamp": "2025-01-01T10:00:00"},
        {"event": "page_view", "page": "/dashboard"}
    ],
    memory_type="episodic",
    retention_strength=1.0,
    max_sequence_length=100
)

# Add memories dynamically
neuron.add_memory({"event": "button_click", "button": "submit"})

# Simulate memory decay
neuron.decay_retention(decay_rate=0.01)

# Strengthen memory through reinforcement
neuron.strengthen_retention(boost=0.1)

# Process activation to retrieve memories
result = neuron.process_activation(
    activation=0.8,
    context={"retrieval_mode": "recent", "max_items": 5}
)
```

### Creating Your Own Neuron Type

See the complete guide in `docs/CUSTOM_NEURON_TYPES.md` for step-by-step instructions on:
1. Creating a custom neuron class
2. Implementing required methods
3. Registering with the type registry
4. Adding API support
5. Testing and documentation

Example usage can be found in `examples/example_memory_neuron_usage.py`.

## Requirements Satisfied

This implementation satisfies the following requirements:

- **1.1**: 3D coordinate space with Vector3D
- **1.3**: 384-dimensional vector storage in Neuron
- **2.1**: Directed synapses with weights (-1.0 to 1.0)
- **10.1**: Tool Neuron type with executable code
- **13.1**: NeuronTypeRegistry for extensibility
- **13.2**: Factory methods for type-agnostic creation
- **14.3**: Object pooling for performance
- **15.1**: Modular structure with clear responsibilities
- **15.2**: No circular dependencies
- **15.3**: Specialized files for each neuron type
- **15.5**: Documented project structure

## Next Steps

Future tasks will implement:
- Compression Engine (Task 2)
- Spatial Index (Task 3)
- Storage Layer (Task 4)
- Query Engine (Task 5)
- Training Engine (Task 6)
- Tool Execution (Task 7)
- Tool Clusters (Task 8)
- REST API (Task 10)
- Python SDK (Task 11)
