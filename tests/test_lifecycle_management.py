"""
Test neuron and synapse lifecycle management functionality.

Tests for Task 9: Fast neuron creation, deletion with cascade, and batch operations.
"""

import time
import numpy as np
from uuid import uuid4

from neuron_system.core import Vector3D, NeuronGraph, Synapse, SynapseType
from neuron_system.neuron_types import KnowledgeNeuron, ToolNeuron


def test_fast_neuron_creation():
    """Test fast neuron creation with automatic positioning."""
    print("Testing fast neuron creation...")
    
    graph = NeuronGraph()
    
    # Test 1: Create neuron with random positioning
    start_time = time.time()
    neuron1 = graph.create_neuron(
        neuron_type="knowledge",
        source_data="Test neuron 1",
        auto_position=True,
        positioning_strategy="random",
        lazy_vector=True
    )
    elapsed_ms = (time.time() - start_time) * 1000
    
    print(f"  Created neuron in {elapsed_ms:.3f}ms")
    print(f"  Neuron ID: {neuron1.id}")
    print(f"  Position: {neuron1.position}")
    print(f"  Vector is None (lazy): {neuron1.vector is None}")
    assert elapsed_ms < 5.0, f"Creation took {elapsed_ms}ms, expected < 5ms"
    assert neuron1.id is not None
    assert neuron1.position is not None
    assert neuron1.vector is None  # Lazy generation
    
    # Test 2: Create neuron with immediate vector
    neuron2 = graph.create_neuron(
        neuron_type="knowledge",
        source_data="Test neuron 2",
        vector=np.random.rand(384),
        lazy_vector=False,
        auto_position=True,
        positioning_strategy="random"
    )
    
    print(f"  Neuron with vector: {neuron2.vector is not None}")
    assert neuron2.vector is not None
    
    # Test 3: Create neuron near similar (with vector)
    neuron3 = graph.create_neuron(
        neuron_type="knowledge",
        source_data="Test neuron 3",
        vector=np.random.rand(384),
        lazy_vector=False,
        auto_position=True,
        positioning_strategy="similar",
        k=2,
        spread=5.0
    )
    
    print(f"  Neuron positioned near similar: {neuron3.position}")
    
    # Test 4: Create neuron in cluster
    neuron4 = graph.create_neuron(
        neuron_type="tool",
        function_signature="test() -> None",
        executable_code="pass",
        auto_position=True,
        positioning_strategy="cluster",
        center=Vector3D(0, 0, 0),
        radius=10.0
    )
    
    print(f"  Tool neuron in cluster: {neuron4.position}")
    assert neuron4.position.distance(Vector3D(0, 0, 0)) <= 10.0
    
    # Test 5: Create neuron on grid
    neuron5 = graph.create_neuron(
        neuron_type="knowledge",
        source_data="Test neuron 5",
        auto_position=True,
        positioning_strategy="grid",
        grid_size=10
    )
    
    print(f"  Neuron on grid: {neuron5.position}")
    
    # Test 6: Verify all neurons are in graph
    assert graph.get_neuron_count() == 5
    print(f"  Total neurons in graph: {graph.get_neuron_count()}")
    
    # Test 7: Verify positions are within bounds
    for neuron in graph.neurons.values():
        assert graph.bounds[0].x <= neuron.position.x <= graph.bounds[1].x
        assert graph.bounds[0].y <= neuron.position.y <= graph.bounds[1].y
        assert graph.bounds[0].z <= neuron.position.z <= graph.bounds[1].z
    
    print("  ✓ Fast neuron creation working\n")


def test_neuron_deletion_cascade():
    """Test neuron deletion with cascade."""
    print("Testing neuron deletion with cascade...")
    
    graph = NeuronGraph()
    
    # Create neurons
    n1 = graph.create_neuron(neuron_type="knowledge", source_data="Neuron 1")
    n2 = graph.create_neuron(neuron_type="knowledge", source_data="Neuron 2")
    n3 = graph.create_neuron(neuron_type="knowledge", source_data="Neuron 3")
    
    print(f"  Created {graph.get_neuron_count()} neurons")
    
    # Create synapses
    s1 = Synapse(source_neuron_id=n1.id, target_neuron_id=n2.id, weight=0.8)
    s2 = Synapse(source_neuron_id=n1.id, target_neuron_id=n3.id, weight=0.6)
    s3 = Synapse(source_neuron_id=n2.id, target_neuron_id=n3.id, weight=0.7)
    
    graph.add_synapse(s1)
    graph.add_synapse(s2)
    graph.add_synapse(s3)
    
    print(f"  Created {graph.get_synapse_count()} synapses")
    assert graph.get_synapse_count() == 3
    
    # Test cascade deletion
    # n1 has 2 outgoing synapses (s1, s2)
    # Deleting n1 should also delete s1 and s2
    result = graph.remove_neuron(n1.id)
    
    print(f"  Deleted neuron n1: {result}")
    print(f"  Remaining neurons: {graph.get_neuron_count()}")
    print(f"  Remaining synapses: {graph.get_synapse_count()}")
    
    assert result is True
    assert graph.get_neuron_count() == 2
    assert graph.get_synapse_count() == 1  # Only s3 remains
    assert graph.get_synapse(s1.id) is None
    assert graph.get_synapse(s2.id) is None
    assert graph.get_synapse(s3.id) is not None
    
    # Test deletion of non-existent neuron
    result = graph.remove_neuron(uuid4())
    print(f"  Delete non-existent neuron: {result}")
    assert result is False
    
    # Test that neuron is removed from graph (spatial index may still contain it until rebuild)
    assert graph.get_neuron(n1.id) is None
    print(f"  Neuron n1 removed from graph: {graph.get_neuron(n1.id) is None}")
    
    print("  ✓ Neuron deletion with cascade working\n")


def test_batch_neuron_creation():
    """Test batch neuron creation for high throughput."""
    print("Testing batch neuron creation...")
    
    graph = NeuronGraph()
    
    # Test 1: Create small batch
    neuron_specs = [
        {"neuron_type": "knowledge", "source_data": f"Neuron {i}"}
        for i in range(10)
    ]
    
    start_time = time.time()
    neurons = graph.batch_create_neurons(
        neuron_specs,
        auto_position=True,
        positioning_strategy="random",
        lazy_vector=True
    )
    elapsed = time.time() - start_time
    
    print(f"  Created {len(neurons)} neurons in {elapsed:.3f}s")
    print(f"  Throughput: {len(neurons) / elapsed:.0f} neurons/second")
    assert len(neurons) == 10
    assert graph.get_neuron_count() == 10
    
    # Test 2: Create large batch (performance test)
    large_batch_specs = [
        {"neuron_type": "knowledge", "source_data": f"Batch neuron {i}"}
        for i in range(1000)
    ]
    
    start_time = time.time()
    large_neurons = graph.batch_create_neurons(
        large_batch_specs,
        auto_position=True,
        positioning_strategy="random",
        lazy_vector=True
    )
    elapsed = time.time() - start_time
    throughput = len(large_neurons) / elapsed
    
    print(f"  Created {len(large_neurons)} neurons in {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.0f} neurons/second")
    assert len(large_neurons) == 1000
    assert graph.get_neuron_count() == 1010
    assert throughput > 1000, f"Throughput {throughput:.0f} < 1000 neurons/second"
    
    # Test 3: Test transaction rollback on error
    invalid_specs = [
        {"neuron_type": "knowledge", "source_data": "Valid 1"},
        {"neuron_type": "invalid_type", "source_data": "Invalid"},  # This will fail
        {"neuron_type": "knowledge", "source_data": "Valid 2"}
    ]
    
    try:
        graph.batch_create_neurons(invalid_specs, use_transaction=True)
        assert False, "Should have raised an error"
    except RuntimeError as e:
        print(f"  Transaction rollback on error: {str(e)[:50]}...")
        # Verify no partial creation
        assert graph.get_neuron_count() == 1010  # Same as before
    
    # Test 4: Test partial success without transaction
    partial_neurons = graph.batch_create_neurons(
        [{"neuron_type": "knowledge", "source_data": "Partial"}],
        use_transaction=False
    )
    print(f"  Partial success created: {len(partial_neurons)} neurons")
    assert len(partial_neurons) == 1
    
    print("  ✓ Batch neuron creation working\n")


def test_batch_synapse_creation():
    """Test batch synapse creation."""
    print("Testing batch synapse creation...")
    
    graph = NeuronGraph()
    
    # Create neurons first
    neurons = graph.batch_create_neurons(
        [{"neuron_type": "knowledge", "source_data": f"N{i}"} for i in range(10)],
        auto_position=True
    )
    
    neuron_ids = [n.id for n in neurons]
    
    # Test 1: Create batch of synapses
    synapse_specs = [
        {
            "source_neuron_id": neuron_ids[i],
            "target_neuron_id": neuron_ids[(i + 1) % 10],
            "weight": 0.5 + (i * 0.05),
            "synapse_type": "KNOWLEDGE"
        }
        for i in range(10)
    ]
    
    start_time = time.time()
    synapses = graph.batch_add_synapses(synapse_specs)
    elapsed = time.time() - start_time
    
    print(f"  Created {len(synapses)} synapses in {elapsed:.3f}s")
    assert len(synapses) == 10
    assert graph.get_synapse_count() == 10
    
    # Test 2: Verify synapse properties
    for i, synapse in enumerate(synapses):
        assert synapse.source_neuron_id == neuron_ids[i]
        assert synapse.target_neuron_id == neuron_ids[(i + 1) % 10]
        assert abs(synapse.weight - (0.5 + i * 0.05)) < 0.001
    
    print(f"  All synapses have correct properties")
    
    # Test 3: Test validation (invalid neuron reference)
    invalid_specs = [
        {
            "source_neuron_id": uuid4(),  # Non-existent neuron
            "target_neuron_id": neuron_ids[0],
            "weight": 0.5
        }
    ]
    
    try:
        graph.batch_add_synapses(invalid_specs, use_transaction=True)
        assert False, "Should have raised an error"
    except (ValueError, RuntimeError) as e:
        print(f"  Validation error caught: {str(e)[:50]}...")
    
    # Test 4: Large batch performance
    large_synapse_specs = [
        {
            "source_neuron_id": neuron_ids[i % 10],
            "target_neuron_id": neuron_ids[(i + 1) % 10],
            "weight": 0.5
        }
        for i in range(100)
    ]
    
    start_time = time.time()
    large_synapses = graph.batch_add_synapses(large_synapse_specs)
    elapsed = time.time() - start_time
    throughput = len(large_synapses) / elapsed
    
    print(f"  Created {len(large_synapses)} synapses in {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.0f} synapses/second")
    assert len(large_synapses) == 100
    
    print("  ✓ Batch synapse creation working\n")


def test_batch_deletion():
    """Test batch deletion operations."""
    print("Testing batch deletion...")
    
    graph = NeuronGraph()
    
    # Create neurons and synapses
    neurons = graph.batch_create_neurons(
        [{"neuron_type": "knowledge", "source_data": f"N{i}"} for i in range(20)],
        auto_position=True
    )
    
    neuron_ids = [n.id for n in neurons]
    
    synapse_specs = [
        {
            "source_neuron_id": neuron_ids[i],
            "target_neuron_id": neuron_ids[(i + 1) % 20],
            "weight": 0.5
        }
        for i in range(20)
    ]
    
    synapses = graph.batch_add_synapses(synapse_specs)
    synapse_ids = [s.id for s in synapses]
    
    print(f"  Initial state: {graph.get_neuron_count()} neurons, {graph.get_synapse_count()} synapses")
    
    # Test 1: Batch delete synapses
    deleted_count = graph.batch_remove_synapses(synapse_ids[:10])
    print(f"  Deleted {deleted_count} synapses")
    assert deleted_count == 10
    assert graph.get_synapse_count() == 10
    
    # Test 2: Batch delete neurons (with cascade)
    deleted_count = graph.batch_remove_neurons(neuron_ids[:5])
    print(f"  Deleted {deleted_count} neurons")
    assert deleted_count == 5
    assert graph.get_neuron_count() == 15
    # Some synapses should be cascade deleted
    print(f"  Remaining synapses after cascade: {graph.get_synapse_count()}")
    
    print("  ✓ Batch deletion working\n")


def test_uuid_pool_integration():
    """Test UUID pool integration with graph."""
    print("Testing UUID pool integration...")
    
    graph = NeuronGraph()
    
    # Check initial pool stats
    initial_stats = graph.uuid_pool.get_stats()
    print(f"  Initial pool size: {initial_stats['pool_size']}")
    print(f"  Initial allocated: {initial_stats['allocated_count']}")
    
    # Create many neurons to test pool refill
    neurons = graph.batch_create_neurons(
        [{"neuron_type": "knowledge", "source_data": f"N{i}"} for i in range(500)],
        auto_position=True
    )
    
    # Check pool stats after creation
    after_stats = graph.uuid_pool.get_stats()
    print(f"  After creation pool size: {after_stats['pool_size']}")
    print(f"  After creation allocated: {after_stats['allocated_count']}")
    
    assert after_stats['allocated_count'] == initial_stats['allocated_count'] + 500
    assert after_stats['pool_size'] >= after_stats['refill_threshold']
    
    print("  ✓ UUID pool integration working\n")


def test_performance_benchmark():
    """Benchmark overall performance."""
    print("Performance Benchmark...")
    
    graph = NeuronGraph()
    
    # Benchmark 1: Single neuron creation speed
    times = []
    for _ in range(100):
        start = time.time()
        graph.create_neuron(
            neuron_type="knowledge",
            source_data="Benchmark",
            lazy_vector=True
        )
        times.append((time.time() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    print(f"  Average single neuron creation: {avg_time:.3f}ms")
    assert avg_time < 2.0, f"Average time {avg_time:.3f}ms exceeds 2ms target"
    
    # Benchmark 2: Batch creation throughput
    start_time = time.time()
    neurons = graph.batch_create_neurons(
        [{"neuron_type": "knowledge", "source_data": f"B{i}"} for i in range(10000)],
        auto_position=True,
        lazy_vector=True
    )
    elapsed = time.time() - start_time
    throughput = len(neurons) / elapsed
    
    print(f"  Batch creation throughput: {throughput:.0f} neurons/second")
    assert throughput > 5000, f"Throughput {throughput:.0f} < 5000 neurons/second"
    
    print("  ✓ Performance benchmarks passed\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Neuron and Synapse Lifecycle Management Tests")
    print("=" * 70 + "\n")
    
    test_fast_neuron_creation()
    test_neuron_deletion_cascade()
    test_batch_neuron_creation()
    test_batch_synapse_creation()
    test_batch_deletion()
    test_uuid_pool_integration()
    test_performance_benchmark()
    
    print("=" * 70)
    print("All lifecycle management tests passed! ✓")
    print("=" * 70)
