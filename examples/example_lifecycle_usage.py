"""
Example usage of neuron and synapse lifecycle management.

Demonstrates:
- Fast neuron creation with automatic positioning
- Lazy vector generation
- Neuron deletion with cascade
- Batch operations for high throughput
"""

import time
import numpy as np

from neuron_system.core import NeuronGraph, Vector3D
# Import neuron types to register them
from neuron_system.neuron_types import KnowledgeNeuron, ToolNeuron


def example_fast_creation():
    """Example: Fast neuron creation."""
    print("=" * 70)
    print("Example 1: Fast Neuron Creation")
    print("=" * 70)
    
    graph = NeuronGraph()
    
    # Create a knowledge neuron with lazy vector generation
    print("\n1. Creating knowledge neuron with lazy vector...")
    start = time.time()
    neuron1 = graph.create_neuron(
        neuron_type="knowledge",
        source_data="Python is a high-level programming language",
        auto_position=True,
        positioning_strategy="random",
        lazy_vector=True  # Vector generated on first query
    )
    elapsed_ms = (time.time() - start) * 1000
    
    print(f"   Created in {elapsed_ms:.3f}ms")
    print(f"   ID: {neuron1.id}")
    print(f"   Position: {neuron1.position}")
    print(f"   Vector is lazy: {neuron1.vector is None}")
    
    # Create a neuron with immediate vector
    print("\n2. Creating neuron with immediate vector...")
    neuron2 = graph.create_neuron(
        neuron_type="knowledge",
        source_data="Machine learning is a subset of AI",
        vector=np.random.rand(384),
        lazy_vector=False,
        auto_position=True,
        positioning_strategy="random"
    )
    print(f"   Vector generated: {neuron2.vector is not None}")
    
    # Create a neuron positioned near similar neurons
    print("\n3. Creating neuron near similar neurons...")
    neuron3 = graph.create_neuron(
        neuron_type="knowledge",
        source_data="Deep learning uses neural networks",
        vector=np.random.rand(384),
        lazy_vector=False,
        auto_position=True,
        positioning_strategy="similar",
        k=2,  # Consider 2 nearest neighbors
        spread=5.0  # Random spread of 5 units
    )
    print(f"   Positioned at: {neuron3.position}")
    print(f"   Distance to neuron2: {neuron3.position.distance(neuron2.position):.2f}")
    
    # Create a tool neuron in a cluster
    print("\n4. Creating tool neuron in cluster...")
    neuron4 = graph.create_neuron(
        neuron_type="tool",
        function_signature="calculate_sum(a: int, b: int) -> int",
        executable_code="return a + b",
        input_schema={"a": "int", "b": "int"},
        auto_position=True,
        positioning_strategy="cluster",
        center=Vector3D(0, 0, 0),
        radius=10.0
    )
    print(f"   Tool neuron at: {neuron4.position}")
    print(f"   Distance from center: {neuron4.position.distance(Vector3D(0, 0, 0)):.2f}")
    
    print(f"\nTotal neurons created: {graph.get_neuron_count()}")


def example_deletion_cascade():
    """Example: Neuron deletion with cascade."""
    print("\n" + "=" * 70)
    print("Example 2: Neuron Deletion with Cascade")
    print("=" * 70)
    
    graph = NeuronGraph()
    
    # Create a small network
    print("\n1. Creating network with 3 neurons and 3 synapses...")
    n1 = graph.create_neuron(neuron_type="knowledge", source_data="Node 1")
    n2 = graph.create_neuron(neuron_type="knowledge", source_data="Node 2")
    n3 = graph.create_neuron(neuron_type="knowledge", source_data="Node 3")
    
    # Create synapses
    from neuron_system.core import Synapse
    s1 = Synapse(source_neuron_id=n1.id, target_neuron_id=n2.id, weight=0.8)
    s2 = Synapse(source_neuron_id=n1.id, target_neuron_id=n3.id, weight=0.6)
    s3 = Synapse(source_neuron_id=n2.id, target_neuron_id=n3.id, weight=0.7)
    
    graph.add_synapse(s1)
    graph.add_synapse(s2)
    graph.add_synapse(s3)
    
    print(f"   Neurons: {graph.get_neuron_count()}")
    print(f"   Synapses: {graph.get_synapse_count()}")
    print(f"   n1 has {len(graph.get_outgoing_synapses(n1.id))} outgoing synapses")
    
    # Delete neuron with cascade
    print("\n2. Deleting n1 (should cascade delete s1 and s2)...")
    graph.remove_neuron(n1.id)
    
    print(f"   Neurons: {graph.get_neuron_count()}")
    print(f"   Synapses: {graph.get_synapse_count()}")
    print(f"   s1 exists: {graph.get_synapse(s1.id) is not None}")
    print(f"   s2 exists: {graph.get_synapse(s2.id) is not None}")
    print(f"   s3 exists: {graph.get_synapse(s3.id) is not None}")


def example_batch_operations():
    """Example: Batch operations for high throughput."""
    print("\n" + "=" * 70)
    print("Example 3: Batch Operations")
    print("=" * 70)
    
    graph = NeuronGraph()
    
    # Batch create neurons
    print("\n1. Batch creating 1000 neurons...")
    neuron_specs = [
        {
            "neuron_type": "knowledge",
            "source_data": f"Knowledge item {i}",
        }
        for i in range(1000)
    ]
    
    start = time.time()
    neurons = graph.batch_create_neurons(
        neuron_specs,
        auto_position=True,
        positioning_strategy="random",
        lazy_vector=True
    )
    elapsed = time.time() - start
    
    print(f"   Created {len(neurons)} neurons in {elapsed:.3f}s")
    print(f"   Throughput: {len(neurons) / elapsed:.0f} neurons/second")
    
    # Batch create synapses
    print("\n2. Batch creating 500 synapses...")
    synapse_specs = [
        {
            "source_neuron_id": neurons[i].id,
            "target_neuron_id": neurons[(i + 1) % len(neurons)].id,
            "weight": 0.5 + (i % 10) * 0.05,
        }
        for i in range(500)
    ]
    
    start = time.time()
    synapses = graph.batch_add_synapses(synapse_specs)
    elapsed = time.time() - start
    
    print(f"   Created {len(synapses)} synapses in {elapsed:.3f}s")
    print(f"   Throughput: {len(synapses) / elapsed:.0f} synapses/second")
    
    # Batch delete
    print("\n3. Batch deleting 100 neurons...")
    neuron_ids_to_delete = [n.id for n in neurons[:100]]
    
    start = time.time()
    deleted_count = graph.batch_remove_neurons(neuron_ids_to_delete)
    elapsed = time.time() - start
    
    print(f"   Deleted {deleted_count} neurons in {elapsed:.3f}s")
    print(f"   Remaining neurons: {graph.get_neuron_count()}")
    print(f"   Remaining synapses: {graph.get_synapse_count()}")


def example_uuid_pool():
    """Example: UUID pool for fast ID allocation."""
    print("\n" + "=" * 70)
    print("Example 4: UUID Pool Performance")
    print("=" * 70)
    
    graph = NeuronGraph()
    
    # Check pool stats
    stats = graph.uuid_pool.get_stats()
    print(f"\n1. Initial pool state:")
    print(f"   Pool size: {stats['pool_size']}")
    print(f"   Allocated: {stats['allocated_count']}")
    print(f"   Refill threshold: {stats['refill_threshold']}")
    
    # Create neurons to trigger refill
    print(f"\n2. Creating 2000 neurons to trigger pool refill...")
    neurons = graph.batch_create_neurons(
        [{"neuron_type": "knowledge", "source_data": f"N{i}"} for i in range(2000)],
        auto_position=True
    )
    
    stats = graph.uuid_pool.get_stats()
    print(f"   Pool size after creation: {stats['pool_size']}")
    print(f"   Total allocated: {stats['allocated_count']}")
    print(f"   Pool automatically refilled: {stats['pool_size'] > stats['refill_threshold']}")


def example_positioning_strategies():
    """Example: Different positioning strategies."""
    print("\n" + "=" * 70)
    print("Example 5: Positioning Strategies")
    print("=" * 70)
    
    graph = NeuronGraph()
    
    # Random positioning
    print("\n1. Random positioning:")
    n1 = graph.create_neuron(
        neuron_type="knowledge",
        source_data="Random 1",
        positioning_strategy="random"
    )
    print(f"   Position: {n1.position}")
    
    # Grid positioning
    print("\n2. Grid positioning:")
    grid_neurons = []
    for i in range(5):
        n = graph.create_neuron(
            neuron_type="knowledge",
            source_data=f"Grid {i}",
            positioning_strategy="grid",
            grid_size=10
        )
        grid_neurons.append(n)
        print(f"   Neuron {i}: {n.position}")
    
    # Cluster positioning
    print("\n3. Cluster positioning (around origin):")
    cluster_neurons = []
    for i in range(5):
        n = graph.create_neuron(
            neuron_type="knowledge",
            source_data=f"Cluster {i}",
            positioning_strategy="cluster",
            center=Vector3D(0, 0, 0),
            radius=5.0
        )
        cluster_neurons.append(n)
        dist = n.position.distance(Vector3D(0, 0, 0))
        print(f"   Neuron {i}: distance from center = {dist:.2f}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Neuron and Synapse Lifecycle Management Examples")
    print("=" * 70)
    
    example_fast_creation()
    example_deletion_cascade()
    example_batch_operations()
    example_uuid_pool()
    example_positioning_strategies()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70 + "\n")
