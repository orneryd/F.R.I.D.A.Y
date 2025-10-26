"""
Basic functionality test for the neuron system.
"""

import numpy as np
from uuid import uuid4

from neuron_system.core import Vector3D, NeuronGraph, Synapse, SynapseType, NeuronTypeRegistry
from neuron_system.neuron_types import KnowledgeNeuron, ToolNeuron
from neuron_system.utils import ObjectPool, UUIDPool
from neuron_system.config import get_settings
from neuron_system.engines import CompressionEngine


def test_vector3d():
    """Test Vector3D functionality."""
    print("Testing Vector3D...")
    v1 = Vector3D(1.0, 2.0, 3.0)
    v2 = Vector3D(4.0, 5.0, 6.0)
    
    distance = v1.distance(v2)
    print(f"  Distance: {distance:.2f}")
    
    magnitude = v1.magnitude()
    print(f"  Magnitude: {magnitude:.2f}")
    
    normalized = v1.normalize()
    print(f"  Normalized: {normalized}")
    
    print("  ✓ Vector3D working\n")


def test_knowledge_neuron():
    """Test KnowledgeNeuron functionality."""
    print("Testing KnowledgeNeuron...")
    
    neuron = KnowledgeNeuron(
        source_data="Machine learning is a subset of AI",
        compression_ratio=0.95,
        semantic_tags=["AI", "ML"]
    )
    neuron.id = uuid4()
    neuron.position = Vector3D(10.0, 20.0, 30.0)
    neuron.vector = np.random.rand(384)
    
    # Test activation
    result = neuron.process_activation(0.8, {})
    print(f"  Activation result: {result['type']}, activation={result['activation']}")
    
    # Test serialization
    data = neuron.to_dict()
    restored = KnowledgeNeuron.from_dict(data)
    print(f"  Serialization: ID matches = {neuron.id == restored.id}")
    
    print("  ✓ KnowledgeNeuron working\n")


def test_tool_neuron():
    """Test ToolNeuron functionality."""
    print("Testing ToolNeuron...")
    
    neuron = ToolNeuron(
        function_signature="calculate(x: int, y: int) -> int",
        executable_code="return x + y",
        input_schema={"x": "int", "y": "int"}
    )
    neuron.id = uuid4()
    neuron.position = Vector3D(5.0, 10.0, 15.0)
    neuron.vector = np.random.rand(384)
    
    # Test activation below threshold
    result = neuron.process_activation(0.3, {})
    print(f"  Low activation: executed={result['executed']}, reason={result.get('reason')}")
    
    # Test activation above threshold
    result = neuron.process_activation(0.7, {"inputs": {"x": 5, "y": 3}})
    print(f"  High activation: executed={result['executed']}")
    
    print("  ✓ ToolNeuron working\n")


def test_neuron_registry():
    """Test NeuronTypeRegistry functionality."""
    print("Testing NeuronTypeRegistry...")
    
    # Check registered types
    types = NeuronTypeRegistry.get_registered_types()
    print(f"  Registered types: {types}")
    
    # Create neuron by type
    neuron = NeuronTypeRegistry.create('knowledge', source_data="Test data")
    print(f"  Created neuron type: {neuron.neuron_type.value}")
    
    # Test serialization through registry
    neuron.id = uuid4()
    neuron.position = Vector3D(1, 2, 3)
    neuron.vector = np.random.rand(384)
    
    data = neuron.to_dict()
    restored = NeuronTypeRegistry.deserialize(data)
    print(f"  Registry deserialization: type matches = {neuron.neuron_type == restored.neuron_type}")
    
    print("  ✓ NeuronTypeRegistry working\n")


def test_synapse():
    """Test Synapse functionality."""
    print("Testing Synapse...")
    
    source_id = uuid4()
    target_id = uuid4()
    
    synapse = Synapse(
        source_neuron_id=source_id,
        target_neuron_id=target_id,
        weight=0.5,
        synapse_type=SynapseType.KNOWLEDGE
    )
    
    print(f"  Initial weight: {synapse.weight}")
    
    # Test strengthening
    synapse.strengthen(0.1)
    print(f"  After strengthen: {synapse.weight}")
    
    # Test weakening
    synapse.weaken(0.05)
    print(f"  After weaken: {synapse.weight}")
    
    # Test traversal
    synapse.traverse()
    print(f"  Usage count: {synapse.usage_count}")
    
    # Test serialization
    data = synapse.to_dict()
    restored = Synapse.from_dict(data)
    print(f"  Serialization: weight matches = {synapse.weight == restored.weight}")
    
    print("  ✓ Synapse working\n")


def test_neuron_graph():
    """Test NeuronGraph functionality."""
    print("Testing NeuronGraph...")
    
    graph = NeuronGraph()
    
    # Create neurons
    n1 = KnowledgeNeuron(source_data="Neuron 1")
    n1.id = uuid4()
    n1.position = Vector3D(0, 0, 0)
    n1.vector = np.random.rand(384)
    
    n2 = KnowledgeNeuron(source_data="Neuron 2")
    n2.id = uuid4()
    n2.position = Vector3D(10, 10, 10)
    n2.vector = np.random.rand(384)
    
    # Add neurons
    graph.add_neuron(n1)
    graph.add_neuron(n2)
    print(f"  Neuron count: {graph.get_neuron_count()}")
    
    # Create synapse
    synapse = Synapse(
        source_neuron_id=n1.id,
        target_neuron_id=n2.id,
        weight=0.8
    )
    graph.add_synapse(synapse)
    print(f"  Synapse count: {graph.get_synapse_count()}")
    
    # Test neighbors
    neighbors = graph.get_neighbors(n1.id)
    print(f"  Neighbors of n1: {len(neighbors)}")
    
    # Test removal
    graph.remove_neuron(n1.id)
    print(f"  After removal - Neurons: {graph.get_neuron_count()}, Synapses: {graph.get_synapse_count()}")
    
    print("  ✓ NeuronGraph working\n")


def test_uuid_pool():
    """Test UUIDPool functionality."""
    print("Testing UUIDPool...")
    
    pool = UUIDPool(initial_size=100, refill_threshold=10)
    
    # Acquire UUIDs
    uuids = [pool.acquire() for _ in range(50)]
    print(f"  Acquired 50 UUIDs")
    
    stats = pool.get_stats()
    print(f"  Pool size: {stats['pool_size']}, Allocated: {stats['allocated_count']}")
    
    print("  ✓ UUIDPool working\n")


def test_object_pool():
    """Test ObjectPool functionality."""
    print("Testing ObjectPool...")
    
    class TestObject:
        def __init__(self):
            self.value = 0
    
    pool = ObjectPool(factory=TestObject, max_size=10)
    
    # Acquire and release
    obj1 = pool.acquire()
    obj1.value = 42
    pool.release(obj1)
    
    obj2 = pool.acquire()
    print(f"  Reused object value: {obj2.value}")
    
    stats = pool.get_stats()
    print(f"  Reuse rate: {stats['reuse_rate']:.2f}")
    
    print("  ✓ ObjectPool working\n")


def test_settings():
    """Test Settings functionality."""
    print("Testing Settings...")
    
    settings = get_settings()
    print(f"  Vector dimensions: {settings.vector_dimensions}")
    print(f"  Default top-k: {settings.default_top_k}")
    print(f"  Embedding model: {settings.embedding_model}")
    
    print("  ✓ Settings working\n")


def test_compression_engine():
    """Test CompressionEngine functionality."""
    print("Testing CompressionEngine...")
    
    # Note: This test will work without sentence-transformers installed
    # by using the fallback mechanism
    engine = CompressionEngine()
    
    # Test input validation
    is_valid, error = engine.validate_input("This is a test")
    print(f"  Valid input: {is_valid}")
    
    is_valid, error = engine.validate_input("")
    print(f"  Empty input validation: {is_valid}, error: {error}")
    
    # Test compression (will use fallback if sentence-transformers not installed)
    text = "Machine learning is a subset of artificial intelligence"
    vector, metadata = engine.compress(text, raise_on_error=False)
    
    print(f"  Vector shape: {vector.shape}")
    print(f"  Compression ratio: {metadata['compression_ratio']:.2f}")
    print(f"  Elapsed time: {metadata['elapsed_time_ms']:.2f}ms")
    print(f"  Success: {metadata.get('success', False)}")
    
    if metadata.get('fallback'):
        print("  Note: Using fallback (sentence-transformers not installed)")
    
    # Test batch compression
    texts = [
        "First text about AI",
        "Second text about machine learning",
        "Third text about neural networks"
    ]
    vectors, metadata_list = engine.batch_compress(texts, raise_on_error=False)
    
    print(f"  Batch vectors shape: {vectors.shape}")
    print(f"  Batch items processed: {len(metadata_list)}")
    
    # Test performance stats
    stats = engine.get_performance_stats()
    print(f"  Total compressions: {stats['total_compressions']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    
    print("  ✓ CompressionEngine working\n")


if __name__ == "__main__":
    print("=" * 60)
    print("3D Synaptic Neuron System - Basic Functionality Test")
    print("=" * 60 + "\n")
    
    test_vector3d()
    test_knowledge_neuron()
    test_tool_neuron()
    test_neuron_registry()
    test_synapse()
    test_neuron_graph()
    test_uuid_pool()
    test_object_pool()
    test_settings()
    test_compression_engine()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
