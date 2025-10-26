# Neuron System Python SDK

The Neuron System Python SDK provides a convenient interface for interacting with the 3D Synaptic Neuron System API.

## Installation

The SDK is included with the neuron_system package. Make sure you have the required dependencies:

```bash
pip install requests
```

## Quick Start

### Basic Usage

```python
from neuron_system.sdk import NeuronSystemClient

# Initialize the client
client = NeuronSystemClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"  # Optional
)

# Check system health
health = client.health_check()
print(f"Status: {health['status']}, Neurons: {health['neuron_count']}")

# Add knowledge to the system
neuron = client.add_knowledge(
    text="Python is a high-level programming language",
    tags=["programming", "python"]
)
print(f"Created neuron: {neuron['id']}")

# Search for knowledge
results = client.search("programming language", limit=5)
for result in results:
    print(f"Score: {result['score']:.2f} - {result['content']}")
```

### Using Context Manager

```python
from neuron_system.sdk import NeuronSystemClient

with NeuronSystemClient(base_url="http://localhost:8000") as client:
    # Client automatically closes when exiting the context
    results = client.search("machine learning")
    print(f"Found {len(results)} results")
```

## Core Concepts

### Neurons

Neurons are the fundamental units of knowledge in the system. There are two main types:

- **Knowledge Neurons**: Store compressed information
- **Tool Neurons**: Execute functions when activated

### Synapses

Synapses are weighted connections between neurons that represent relationships and enable activation propagation.

### Queries

Queries search the neuron network by:
1. Compressing the query text to a vector
2. Finding spatially nearby neurons
3. Propagating activation through synapses
4. Returning the most activated neurons

## API Reference

### Client Initialization

```python
client = NeuronSystemClient(
    base_url="http://localhost:8000",  # API base URL
    api_key=None,                       # Optional API key
    timeout=30                          # Request timeout in seconds
)
```

### High-Level Methods

These methods provide simplified interfaces for common operations:

#### add_knowledge()

Add knowledge to the system:

```python
neuron = client.add_knowledge(
    text="Machine learning is a subset of AI",
    tags=["AI", "ML"],
    metadata={"source": "textbook"}
)
```

#### add_tool()

Add a tool to the system:

```python
result = client.add_tool(
    name="calculate_sum",
    description="Calculate the sum of two numbers",
    code="result = a + b",
    input_schema={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    },
    output_schema={"type": "number"}
)
tool_id = result['details']['tool_neuron_id']
```

#### search()

Search for knowledge:

```python
results = client.search(
    text="What is machine learning?",
    limit=10,
    depth=3
)

for result in results:
    print(f"ID: {result['id']}")
    print(f"Score: {result['score']}")
    print(f"Content: {result['content']}")
    print(f"Tags: {result['tags']}")
```

#### train()

Train a neuron with new knowledge:

```python
result = client.train(
    neuron_id="123e4567-e89b-12d3-a456-426614174000",
    new_knowledge="Python 3.12 was released in October 2023",
    learning_rate=0.2
)
```

#### connect()

Connect two neurons:

```python
synapse = client.connect(
    from_neuron_id="123e4567-...",
    to_neuron_id="987f6543-...",
    strength=0.8
)
```

#### strengthen_connection() / weaken_connection()

Modify connection strength:

```python
# Strengthen
result = client.strengthen_connection(
    synapse_id="123e4567-...",
    amount=0.1
)

# Weaken
result = client.weaken_connection(
    synapse_id="123e4567-...",
    amount=0.05
)
```

### Low-Level Methods

For more control, use the low-level API methods:

#### Neuron Operations

```python
# Create a neuron
neuron = client.create_neuron(
    neuron_type="knowledge",
    source_data="Some knowledge",
    semantic_tags=["tag1", "tag2"],
    position=(10.0, 20.0, 30.0)  # Optional 3D position
)

# Get a neuron
neuron = client.get_neuron(neuron_id="123e4567-...")

# Delete a neuron
result = client.delete_neuron(neuron_id="123e4567-...")

# Batch create neurons
result = client.create_neurons_batch([
    {"neuron_type": "knowledge", "source_data": "Data 1"},
    {"neuron_type": "knowledge", "source_data": "Data 2"}
])
```

#### Synapse Operations

```python
# Create a synapse
synapse = client.create_synapse(
    source_neuron_id="123e4567-...",
    target_neuron_id="987f6543-...",
    weight=0.5,
    synapse_type="KNOWLEDGE"
)

# Get a synapse
synapse = client.get_synapse(synapse_id="123e4567-...")

# Query synapses
synapses = client.query_synapses(
    source_neuron_id="123e4567-...",
    min_weight=0.5
)

# Delete a synapse
result = client.delete_synapse(synapse_id="123e4567-...")
```

#### Query Operations

```python
# Knowledge query
results = client.query(
    query_text="What is Python?",
    top_k=10,
    propagation_depth=3,
    neuron_type_filter="knowledge"
)

# Spatial query
results = client.spatial_query(
    center=(0.0, 0.0, 0.0),
    radius=10.0,
    neuron_type_filter="knowledge"
)

# Get neighbors
neighbors = client.get_neighbors(neuron_id="123e4567-...")
```

#### Training Operations

```python
# Adjust neuron vector
result = client.adjust_neuron(
    neuron_id="123e4567-...",
    target_text="New knowledge",
    learning_rate=0.1
)

# Or use a target vector directly
result = client.adjust_neuron(
    neuron_id="123e4567-...",
    target_vector=[0.1, 0.2, ...],  # 384 dimensions
    learning_rate=0.1
)

# Adjust synapse weight
result = client.adjust_synapse(
    synapse_id="123e4567-...",
    operation="strengthen",  # or "weaken" or "set"
    delta=0.1
)

# Create tool neuron
result = client.create_tool_neuron(
    description="Calculate factorial",
    function_signature="factorial(n: int)",
    executable_code="""
import math
result = math.factorial(n)
""",
    input_schema={
        "type": "object",
        "properties": {"n": {"type": "integer", "minimum": 0}}
    },
    output_schema={"type": "integer"}
)
```

## Error Handling

The SDK provides specific exception types for different error scenarios:

```python
from neuron_system.sdk import (
    NeuronSystemClient,
    ConnectionError,
    AuthenticationError,
    NotFoundError,
    ValidationError,
    ServerError,
    TimeoutError
)

client = NeuronSystemClient(base_url="http://localhost:8000")

try:
    neuron = client.get_neuron("invalid-id")
except NotFoundError as e:
    print(f"Neuron not found: {e}")
except ValidationError as e:
    print(f"Invalid request: {e}")
except ConnectionError as e:
    print(f"Connection failed: {e}")
except TimeoutError as e:
    print(f"Request timed out: {e}")
except ServerError as e:
    print(f"Server error: {e}")
```

### Exception Hierarchy

- `NeuronSystemError` (base exception)
  - `ConnectionError` - Connection to API failed
  - `AuthenticationError` - Authentication failed (401)
  - `NotFoundError` - Resource not found (404)
  - `ValidationError` - Request validation failed (400)
  - `ServerError` - Server error (500+)
  - `TimeoutError` - Request timed out

## Examples

### Example 1: Building a Knowledge Base

```python
from neuron_system.sdk import NeuronSystemClient

client = NeuronSystemClient(base_url="http://localhost:8000")

# Add multiple pieces of knowledge
knowledge_items = [
    "Python is a high-level programming language",
    "Machine learning is a subset of artificial intelligence",
    "Neural networks are inspired by biological neurons",
    "Deep learning uses multiple layers of neural networks"
]

neuron_ids = []
for text in knowledge_items:
    neuron = client.add_knowledge(text, tags=["AI", "ML"])
    neuron_ids.append(neuron['id'])
    print(f"Added: {text}")

# Connect related concepts
for i in range(len(neuron_ids) - 1):
    synapse = client.connect(
        from_neuron_id=neuron_ids[i],
        to_neuron_id=neuron_ids[i + 1],
        strength=0.7
    )
    print(f"Connected neurons {i} -> {i+1}")

# Search the knowledge base
results = client.search("neural networks", limit=3)
print(f"\nSearch results for 'neural networks':")
for result in results:
    print(f"  Score: {result['score']:.2f} - {result['content']}")
```

### Example 2: Creating and Using Tools

```python
from neuron_system.sdk import NeuronSystemClient

client = NeuronSystemClient(base_url="http://localhost:8000")

# Create a calculator tool
result = client.add_tool(
    name="multiply",
    description="Multiply two numbers",
    code="result = x * y",
    input_schema={
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"}
        },
        "required": ["x", "y"]
    },
    output_schema={"type": "number"}
)

tool_id = result['details']['tool_neuron_id']
print(f"Created tool neuron: {tool_id}")

# Add knowledge about multiplication
knowledge = client.add_knowledge(
    text="Multiplication is a mathematical operation",
    tags=["math", "arithmetic"]
)

# Connect the knowledge to the tool
synapse = client.connect(
    from_neuron_id=knowledge['id'],
    to_neuron_id=tool_id,
    strength=0.9
)

print("Tool connected to knowledge base")
```

### Example 3: Training and Updating Knowledge

```python
from neuron_system.sdk import NeuronSystemClient

client = NeuronSystemClient(base_url="http://localhost:8000")

# Add initial knowledge
neuron = client.add_knowledge(
    text="Python 3.11 is the latest version",
    tags=["python", "version"]
)

print(f"Initial knowledge: {neuron['source_data']}")

# Update the knowledge
result = client.train(
    neuron_id=neuron['id'],
    new_knowledge="Python 3.12 is the latest version",
    learning_rate=0.5  # Higher learning rate for significant update
)

print(f"Training result: {result['message']}")

# Verify the update
updated_neuron = client.get_neuron(neuron['id'])
print(f"Updated knowledge stored in neuron")
```

### Example 4: Exploring Network Connections

```python
from neuron_system.sdk import NeuronSystemClient

client = NeuronSystemClient(base_url="http://localhost:8000")

# Get network statistics
stats = client.get_network_stats()
print(f"Network has {stats['neuron_count']} neurons and {stats['synapse_count']} synapses")

# Create a neuron
neuron = client.add_knowledge(
    text="Graph databases store data as nodes and edges",
    tags=["database", "graph"]
)

# Get its neighbors
neighbors = client.get_neighbors(neuron['id'])
print(f"Neuron has {neighbors['count']} neighbors")

for neighbor_data in neighbors['neighbors']:
    synapse = neighbor_data['synapse']
    neighbor = neighbor_data['neuron']
    print(f"  -> {neighbor['id']} (weight: {synapse['weight']})")
```

## Best Practices

### 1. Use Context Managers

Always use context managers to ensure proper cleanup:

```python
with NeuronSystemClient(base_url="http://localhost:8000") as client:
    # Your code here
    pass
# Client automatically closed
```

### 2. Handle Errors Gracefully

Always wrap API calls in try-except blocks:

```python
try:
    results = client.search("query")
except NotFoundError:
    print("No results found")
except ConnectionError:
    print("Could not connect to API")
```

### 3. Use High-Level Methods

Prefer high-level convenience methods for common operations:

```python
# Good
neuron = client.add_knowledge("Some text")

# Also works, but more verbose
neuron = client.create_neuron(
    neuron_type="knowledge",
    source_data="Some text"
)
```

### 4. Batch Operations

Use batch operations for better performance:

```python
# Create multiple neurons at once
result = client.create_neurons_batch([
    {"neuron_type": "knowledge", "source_data": f"Data {i}"}
    for i in range(100)
])
```

### 5. Set Appropriate Timeouts

Adjust timeout based on your use case:

```python
# For long-running operations
client = NeuronSystemClient(
    base_url="http://localhost:8000",
    timeout=60  # 60 seconds
)
```

## Troubleshooting

### Connection Issues

If you can't connect to the API:

1. Verify the API is running: `curl http://localhost:8000/health`
2. Check the base_url is correct
3. Ensure no firewall is blocking the connection

### Authentication Errors

If you get authentication errors:

1. Verify your API key is correct
2. Check if the API requires authentication
3. Ensure the API key header is being sent

### Timeout Errors

If requests are timing out:

1. Increase the timeout value
2. Check if the API is overloaded
3. Verify network connectivity

### Validation Errors

If you get validation errors:

1. Check the request parameters match the API requirements
2. Verify data types are correct
3. Ensure required fields are provided

## Support

For issues or questions:

- Check the API documentation at `/docs` endpoint
- Review the examples in this guide
- Check the error message for specific details
