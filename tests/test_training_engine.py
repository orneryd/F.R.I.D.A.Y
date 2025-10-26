"""
Test script for Training Engine functionality.
"""

import numpy as np
from uuid import uuid4
from datetime import datetime, timedelta

from neuron_system.core.graph import NeuronGraph
from neuron_system.core.vector3d import Vector3D
from neuron_system.core.synapse import Synapse, SynapseType
from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
from neuron_system.engines.training import TrainingEngine


def test_training_engine():
    """Test basic Training Engine functionality."""
    print("Testing Training Engine...")
    
    # Create graph
    graph = NeuronGraph()
    
    # Create test neurons
    neuron1 = KnowledgeNeuron(
        source_data="Test knowledge 1",
        compression_ratio=100.0
    )
    neuron1.id = uuid4()
    neuron1.position = Vector3D(0, 0, 0)
    neuron1.vector = np.random.randn(384)
    neuron1.created_at = datetime.now()
    neuron1.modified_at = datetime.now()
    
    neuron2 = KnowledgeNeuron(
        source_data="Test knowledge 2",
        compression_ratio=100.0
    )
    neuron2.id = uuid4()
    neuron2.position = Vector3D(1, 1, 1)
    neuron2.vector = np.random.randn(384)
    neuron2.created_at = datetime.now()
    neuron2.modified_at = datetime.now()
    
    # Add neurons to graph
    graph.add_neuron(neuron1)
    graph.add_neuron(neuron2)
    
    # Create synapse
    synapse = Synapse(
        id=uuid4(),
        source_neuron_id=neuron1.id,
        target_neuron_id=neuron2.id,
        weight=0.5,
        synapse_type=SynapseType.KNOWLEDGE
    )
    graph.add_synapse(synapse)
    
    # Initialize training engine
    engine = TrainingEngine(graph, learning_rate=0.1, decay_rate=0.001)
    
    print(f"✓ Created graph with {graph.get_neuron_count()} neurons and {graph.get_synapse_count()} synapses")
    
    # Test 1: Adjust neuron vector
    print("\n1. Testing neuron vector adjustment...")
    original_vector = neuron1.vector.copy()
    target_vector = np.random.randn(384)
    
    success = engine.adjust_neuron(neuron1.id, target_vector, learning_rate=0.2)
    assert success, "Failed to adjust neuron"
    
    # Verify vector changed
    assert not np.array_equal(neuron1.vector, original_vector), "Vector should have changed"
    print(f"✓ Neuron vector adjusted successfully")
    print(f"  Original norm: {np.linalg.norm(original_vector):.4f}")
    print(f"  New norm: {np.linalg.norm(neuron1.vector):.4f}")
    
    # Test 2: Strengthen synapse
    print("\n2. Testing synapse strengthening...")
    original_weight = synapse.weight
    
    success = engine.strengthen_synapse(synapse.id, delta=0.1)
    assert success, "Failed to strengthen synapse"
    assert synapse.weight > original_weight, "Weight should have increased"
    assert synapse.weight <= 1.0, "Weight should not exceed 1.0"
    
    print(f"✓ Synapse strengthened: {original_weight:.4f} -> {synapse.weight:.4f}")
    
    # Test 3: Weaken synapse
    print("\n3. Testing synapse weakening...")
    original_weight = synapse.weight
    
    success = engine.weaken_synapse(synapse.id, delta=0.05)
    assert success, "Failed to weaken synapse"
    assert synapse.weight < original_weight, "Weight should have decreased"
    
    print(f"✓ Synapse weakened: {original_weight:.4f} -> {synapse.weight:.4f}")
    
    # Test 4: Usage-based learning
    print("\n4. Testing usage-based learning...")
    original_weight = synapse.weight
    original_usage = synapse.usage_count
    
    success = engine.apply_usage_based_learning(synapse.id, strengthen_delta=0.01)
    assert success, "Failed to apply usage-based learning"
    assert synapse.usage_count > original_usage, "Usage count should have increased"
    assert synapse.weight > original_weight, "Weight should have increased"
    
    print(f"✓ Usage-based learning applied")
    print(f"  Usage count: {original_usage} -> {synapse.usage_count}")
    print(f"  Weight: {original_weight:.4f} -> {synapse.weight:.4f}")
    
    # Test 5: Operation logging
    print("\n5. Testing operation logging...")
    log = engine.get_operation_log()
    assert len(log) > 0, "Operation log should not be empty"
    
    print(f"✓ Operation log contains {len(log)} operations")
    for op in log[:3]:  # Show first 3 operations
        print(f"  - {op.operation_type} on {op.target_id} at {op.timestamp}")
    
    # Test 6: Rollback operation
    print("\n6. Testing operation rollback...")
    # Get the last operation
    last_op = log[-1]
    
    # Store current state
    if last_op.operation_type == "strengthen_synapse":
        current_weight = synapse.weight
        
        # Rollback
        success = engine.rollback_operation(last_op.operation_id)
        assert success, "Failed to rollback operation"
        
        # Verify state was restored
        assert synapse.weight != current_weight, "Weight should have been restored"
        print(f"✓ Operation rolled back successfully")
        print(f"  Weight restored: {current_weight:.4f} -> {synapse.weight:.4f}")
    
    # Test 7: Transaction support
    print("\n7. Testing transaction support...")
    with engine.begin_transaction() as txn:
        # Perform multiple operations
        txn.strengthen_synapse(synapse.id, delta=0.05)
        txn.strengthen_synapse(synapse.id, delta=0.05)
        
        # Commit transaction
        txn.commit()
    
    print(f"✓ Transaction committed successfully")
    
    # Test 8: Time-based decay
    print("\n8. Testing time-based decay...")
    # Set last_traversed to old time
    synapse.last_traversed = datetime.now() - timedelta(days=2)
    original_weight = synapse.weight
    
    stats = engine.apply_time_based_decay(time_threshold_seconds=86400, decay_delta=0.01)
    
    print(f"✓ Time-based decay applied")
    print(f"  Decayed: {stats['decayed_count']} synapses")
    print(f"  Deleted: {stats['deleted_count']} synapses")
    
    # Test 9: Statistics
    print("\n9. Testing statistics...")
    stats = engine.get_stats()
    
    print(f"✓ Training statistics:")
    print(f"  Total operations: {stats['total_operations']}")
    print(f"  Learning rate: {stats['learning_rate']}")
    print(f"  Decay rate: {stats['decay_rate']}")
    print(f"  Operation counts: {stats['operation_counts']}")
    
    # Test 10: Validation
    print("\n10. Testing operation validation...")
    is_valid, error = engine.validate_operation(
        "adjust_neuron",
        neuron1.id,
        target_vector=np.random.randn(384)
    )
    assert is_valid, f"Validation should pass: {error}"
    
    is_valid, error = engine.validate_operation(
        "adjust_neuron",
        uuid4(),  # Non-existent neuron
        target_vector=np.random.randn(384)
    )
    assert not is_valid, "Validation should fail for non-existent neuron"
    
    print(f"✓ Operation validation working correctly")
    
    print("\n" + "="*60)
    print("All Training Engine tests passed! ✓")
    print("="*60)


if __name__ == "__main__":
    test_training_engine()
