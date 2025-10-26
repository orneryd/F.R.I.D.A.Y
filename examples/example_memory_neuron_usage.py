"""
Example usage of MemoryNeuron - Custom Neuron Type Extension

This example demonstrates how to use the MemoryNeuron custom neuron type
and serves as a reference for implementing your own custom neuron types.

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neuron_system.core.graph import NeuronGraph
from neuron_system.core.vector3d import Vector3D
from neuron_system.core.synapse import Synapse, SynapseType
from neuron_system.neuron_types.memory_neuron import MemoryNeuron
from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
from neuron_system.engines.compression import CompressionEngine
from neuron_system.engines.query import QueryEngine


def example_1_basic_memory_neuron():
    """Example 1: Creating and using a basic MemoryNeuron"""
    print("=" * 60)
    print("Example 1: Basic MemoryNeuron Creation")
    print("=" * 60)
    
    # Create a memory neuron for storing user activity
    memory_neuron = MemoryNeuron(
        sequence_data=[
            {
                "event": "user_login",
                "user_id": "user_123",
                "timestamp": datetime.now().isoformat()
            },
            {
                "event": "page_view",
                "page": "/dashboard",
                "timestamp": (datetime.now() + timedelta(minutes=1)).isoformat()
            },
            {
                "event": "button_click",
                "button": "export_data",
                "timestamp": (datetime.now() + timedelta(minutes=2)).isoformat()
            }
        ],
        memory_type="episodic",
        retention_strength=1.0,
        max_sequence_length=100
    )
    
    print(f"Created MemoryNeuron: {memory_neuron.id}")
    print(f"Memory Type: {memory_neuron.memory_type}")
    print(f"Retention Strength: {memory_neuron.retention_strength}")
    print(f"Number of memories: {len(memory_neuron.sequence_data)}")
    print(f"Temporal Index: {memory_neuron.temporal_index}")
    
    # Process activation to retrieve memories
    result = memory_neuron.process_activation(
        activation=0.8,
        context={"retrieval_mode": "recent", "max_items": 5}
    )
    
    print(f"\nActivation Result:")
    print(f"  Type: {result['type']}")
    print(f"  Retrieved {result['retrieved_count']} memories")
    print(f"  Effective Activation: {result['effective_activation']:.2f}")
    
    for i, memory in enumerate(result['sequence'], 1):
        print(f"  Memory {i}: {memory['event']}")
    
    print()


def example_2_adding_memories():
    """Example 2: Adding memories dynamically"""
    print("=" * 60)
    print("Example 2: Adding Memories Dynamically")
    print("=" * 60)
    
    # Create an empty memory neuron
    memory_neuron = MemoryNeuron(
        memory_type="episodic",
        max_sequence_length=5  # Small limit for demonstration
    )
    
    print(f"Created empty MemoryNeuron: {memory_neuron.id}")
    print(f"Initial memories: {len(memory_neuron.sequence_data)}")
    
    # Add memories one by one
    events = [
        {"event": "session_start", "session_id": "sess_001"},
        {"event": "search_query", "query": "machine learning"},
        {"event": "result_click", "result_id": "doc_42"},
        {"event": "bookmark_added", "doc_id": "doc_42"},
        {"event": "session_end", "duration_minutes": 15},
        {"event": "new_session", "session_id": "sess_002"},  # This will push out oldest
    ]
    
    for event in events:
        memory_neuron.add_memory(event)
        print(f"Added memory: {event['event']} (total: {len(memory_neuron.sequence_data)})")
    
    print(f"\nFinal memory count: {len(memory_neuron.sequence_data)}")
    print(f"Max sequence length: {memory_neuron.max_sequence_length}")
    print("Note: Oldest memory was removed due to max_sequence_length limit")
    
    # Show remaining memories
    print("\nRemaining memories:")
    for i, memory in enumerate(memory_neuron.sequence_data, 1):
        print(f"  {i}. {memory['event']}")
    
    print()


def example_3_memory_decay():
    """Example 3: Memory retention and decay"""
    print("=" * 60)
    print("Example 3: Memory Retention and Decay")
    print("=" * 60)
    
    # Create a memory neuron
    memory_neuron = MemoryNeuron(
        sequence_data=[
            {"event": "important_event", "data": "critical information"}
        ],
        retention_strength=1.0
    )
    
    print(f"Initial retention strength: {memory_neuron.retention_strength}")
    
    # Simulate memory decay over time
    print("\nSimulating memory decay:")
    for i in range(5):
        memory_neuron.decay_retention(decay_rate=0.15)
        print(f"  After decay {i+1}: {memory_neuron.retention_strength:.2f}")
    
    # Strengthen memory through reinforcement
    print("\nReinforcing memory:")
    memory_neuron.strengthen_retention(boost=0.3)
    print(f"  After reinforcement: {memory_neuron.retention_strength:.2f}")
    
    # Show how retention affects activation
    print("\nActivation with different retention levels:")
    for retention in [1.0, 0.7, 0.4, 0.1]:
        memory_neuron.retention_strength = retention
        result = memory_neuron.process_activation(
            activation=0.8,
            context={"retrieval_mode": "full", "max_items": 10}
        )
        print(f"  Retention {retention:.1f}: Effective activation = {result['effective_activation']:.2f}")
    
    print()


def example_4_integration_with_graph():
    """Example 4: Integrating MemoryNeuron with NeuronGraph"""
    print("=" * 60)
    print("Example 4: Integration with NeuronGraph")
    print("=" * 60)
    
    # Create graph and compression engine
    graph = NeuronGraph()
    compression_engine = CompressionEngine()
    
    # Create a memory neuron
    memory_neuron = graph.create_neuron(
        neuron_type="memory",
        sequence_data=[
            {"event": "user_query", "query": "What is machine learning?"},
            {"event": "user_query", "query": "Explain neural networks"},
        ],
        memory_type="episodic"
    )
    
    print(f"Created MemoryNeuron in graph: {memory_neuron.id}")
    
    # Create knowledge neurons
    knowledge_1 = graph.create_neuron(
        neuron_type="knowledge",
        source_data="Machine learning is a subset of artificial intelligence",
        semantic_tags=["AI", "ML"]
    )
    
    knowledge_2 = graph.create_neuron(
        neuron_type="knowledge",
        source_data="Neural networks are computing systems inspired by biological neural networks",
        semantic_tags=["neural networks", "deep learning"]
    )
    
    print(f"Created KnowledgeNeuron 1: {knowledge_1.id}")
    print(f"Created KnowledgeNeuron 2: {knowledge_2.id}")
    
    # Connect memory neuron to knowledge neurons
    synapse_1 = Synapse(
        source_neuron_id=memory_neuron.id,
        target_neuron_id=knowledge_1.id,
        weight=0.8,
        synapse_type=SynapseType.KNOWLEDGE
    )
    
    synapse_2 = Synapse(
        source_neuron_id=memory_neuron.id,
        target_neuron_id=knowledge_2.id,
        weight=0.7,
        synapse_type=SynapseType.KNOWLEDGE
    )
    
    graph.add_synapse(synapse_1)
    graph.add_synapse(synapse_2)
    
    print(f"\nCreated synapses connecting memory to knowledge neurons")
    print(f"Total neurons in graph: {len(graph.neurons)}")
    print(f"Total synapses in graph: {len(graph.synapses)}")
    
    # Get neighbors
    neighbors = graph.get_neighbors(memory_neuron.id)
    print(f"\nMemory neuron has {len(neighbors)} connected neurons:")
    for synapse, neuron in neighbors:
        print(f"  -> {neuron.neuron_type.value} neuron (weight: {synapse.weight})")
    
    print()


def example_5_serialization():
    """Example 5: Serialization and deserialization"""
    print("=" * 60)
    print("Example 5: Serialization and Deserialization")
    print("=" * 60)
    
    # Create a memory neuron with complex data
    original = MemoryNeuron(
        sequence_data=[
            {
                "event": "complex_event",
                "nested_data": {
                    "key1": "value1",
                    "key2": [1, 2, 3]
                },
                "timestamp": datetime.now().isoformat()
            }
        ],
        memory_type="semantic",
        retention_strength=0.85,
        max_sequence_length=50
    )
    
    print(f"Original MemoryNeuron: {original.id}")
    print(f"  Memory type: {original.memory_type}")
    print(f"  Retention: {original.retention_strength}")
    print(f"  Sequence length: {len(original.sequence_data)}")
    
    # Serialize to dictionary
    data = original.to_dict()
    print(f"\nSerialized to dictionary with {len(data)} keys")
    print(f"  Keys: {list(data.keys())}")
    
    # Deserialize back
    restored = MemoryNeuron.from_dict(data)
    print(f"\nRestored MemoryNeuron: {restored.id}")
    print(f"  Memory type: {restored.memory_type}")
    print(f"  Retention: {restored.retention_strength}")
    print(f"  Sequence length: {len(restored.sequence_data)}")
    
    # Verify data integrity
    assert str(original.id) == str(restored.id)
    assert original.memory_type == restored.memory_type
    assert original.retention_strength == restored.retention_strength
    assert len(original.sequence_data) == len(restored.sequence_data)
    
    print("\n✓ Serialization/deserialization successful!")
    print()


def example_6_different_retrieval_modes():
    """Example 6: Different memory retrieval modes"""
    print("=" * 60)
    print("Example 6: Different Memory Retrieval Modes")
    print("=" * 60)
    
    # Create a memory neuron with various events
    memory_neuron = MemoryNeuron(
        sequence_data=[
            {"event": "login", "priority": "low", "timestamp": "2025-01-01T08:00:00"},
            {"event": "data_access", "priority": "high", "timestamp": "2025-01-01T09:00:00"},
            {"event": "logout", "priority": "low", "timestamp": "2025-01-01T10:00:00"},
            {"event": "login", "priority": "low", "timestamp": "2025-01-01T14:00:00"},
            {"event": "critical_action", "priority": "critical", "timestamp": "2025-01-01T15:00:00"},
            {"event": "logout", "priority": "low", "timestamp": "2025-01-01T16:00:00"},
        ],
        memory_type="episodic"
    )
    
    print(f"Created MemoryNeuron with {len(memory_neuron.sequence_data)} events")
    
    # Test different retrieval modes
    modes = ["full", "recent", "relevant"]
    
    for mode in modes:
        print(f"\n--- Retrieval Mode: {mode} ---")
        result = memory_neuron.process_activation(
            activation=0.9,
            context={"retrieval_mode": mode, "max_items": 3}
        )
        
        print(f"Retrieved {result['retrieved_count']} memories:")
        for i, memory in enumerate(result['sequence'], 1):
            print(f"  {i}. {memory['event']} (priority: {memory['priority']})")
    
    print()


def example_7_clearing_old_memories():
    """Example 7: Clearing old memories"""
    print("=" * 60)
    print("Example 7: Clearing Old Memories")
    print("=" * 60)
    
    # Create memory neuron with timestamped events
    base_time = datetime.now() - timedelta(days=10)
    
    memory_neuron = MemoryNeuron(
        sequence_data=[
            {"event": "old_event_1", "timestamp": (base_time + timedelta(days=0)).isoformat()},
            {"event": "old_event_2", "timestamp": (base_time + timedelta(days=2)).isoformat()},
            {"event": "old_event_3", "timestamp": (base_time + timedelta(days=4)).isoformat()},
            {"event": "recent_event_1", "timestamp": (base_time + timedelta(days=8)).isoformat()},
            {"event": "recent_event_2", "timestamp": (base_time + timedelta(days=9)).isoformat()},
        ],
        memory_type="episodic"
    )
    
    print(f"Initial memory count: {len(memory_neuron.sequence_data)}")
    print("Memories:")
    for memory in memory_neuron.sequence_data:
        print(f"  - {memory['event']}")
    
    # Clear memories older than 5 days
    cutoff_time = base_time + timedelta(days=5)
    removed_count = memory_neuron.clear_old_memories(cutoff_time)
    
    print(f"\nCleared {removed_count} old memories (before {cutoff_time.date()})")
    print(f"Remaining memory count: {len(memory_neuron.sequence_data)}")
    print("Remaining memories:")
    for memory in memory_neuron.sequence_data:
        print(f"  - {memory['event']}")
    
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("MemoryNeuron Custom Neuron Type Examples")
    print("Demonstrating Custom Neuron Type Extension")
    print("=" * 60 + "\n")
    
    try:
        example_1_basic_memory_neuron()
        example_2_adding_memories()
        example_3_memory_decay()
        example_4_integration_with_graph()
        example_5_serialization()
        example_6_different_retrieval_modes()
        example_7_clearing_old_memories()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
