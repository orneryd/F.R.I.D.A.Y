"""
Test memory persistence - verify memories are saved and counted correctly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel
from neuron_system.neuron_types.memory_neuron import MemoryManager, MemoryNeuron


def test_memory_persistence():
    print("=" * 70)
    print("TESTING MEMORY PERSISTENCE")
    print("=" * 70)
    print()
    
    settings = Settings(
        database_path="comprehensive_ai.db",
        spatial_bounds_min=(-500.0, -500.0, -500.0),
        spatial_bounds_max=(500.0, 500.0, 500.0)
    )
    
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        language_model = LanguageModel(
            container.graph,
            container.compression_engine,
            container.query_engine,
            container.training_engine
        )
        
        memory_manager = MemoryManager(
            container.graph,
            container.compression_engine
        )
        
        # Get initial stats
        print("Initial Statistics:")
        stats = language_model.get_statistics()
        print(f"  Total neurons: {stats['total_neurons']:,}")
        print(f"  Memory neurons: {stats['memory_neurons']:,}")
        print()
        
        initial_memory_count = stats['memory_neurons']
        
        # Create some memories
        print("Creating 5 test memories...")
        memory_ids = []
        
        for i in range(5):
            memory_id = memory_manager.create_memory(
                content=f"Test memory {i+1}: This is a test memory for persistence",
                memory_type="short-term",
                context={'test': True, 'index': i},
                importance=0.5 + (i * 0.1)
            )
            memory_ids.append(memory_id)
            print(f"  Created memory {i+1}: {memory_id}")
        
        print()
        
        # Check if memories are in graph
        print("Verifying memories in graph...")
        for i, memory_id in enumerate(memory_ids, 1):
            neuron = container.graph.get_neuron(memory_id)
            if neuron:
                print(f"  Memory {i}: Found in graph [OK]")
                print(f"    Type: {type(neuron).__name__}")
                print(f"    Is MemoryNeuron: {isinstance(neuron, MemoryNeuron)}")
                print(f"    Content: {neuron.source_data[:50]}...")
            else:
                print(f"  Memory {i}: NOT FOUND [FAIL]")
        
        print()
        
        # Get updated stats
        print("Updated Statistics:")
        stats = language_model.get_statistics()
        print(f"  Total neurons: {stats['total_neurons']:,}")
        print(f"  Memory neurons: {stats['memory_neurons']:,}")
        print(f"  Change: +{stats['memory_neurons'] - initial_memory_count}")
        print()
        
        # Verify count
        if stats['memory_neurons'] == initial_memory_count + 5:
            print("[OK] Memory count is correct!")
        else:
            print(f"[FAIL] Memory count is WRONG!")
            print(f"  Expected: {initial_memory_count + 5}")
            print(f"  Got: {stats['memory_neurons']}")
        
        print()
        
        # Test memory manager statistics
        print("Memory Manager Statistics:")
        mem_stats = memory_manager.get_statistics()
        print(f"  Total memories: {mem_stats['total_memories']}")
        print(f"  By type: {mem_stats['by_type']}")
        print(f"  Avg importance: {mem_stats['average_importance']:.2f}")
        print()
        
        # Save to database
        print("Saving to database...")
        
        # Explicitly save memory neurons using create (for new neurons)
        if hasattr(container.graph, 'neuron_store'):
            memory_neurons = [
                n for n in container.graph.neurons.values()
                if isinstance(n, MemoryNeuron)
            ]
            print(f"  Saving {len(memory_neurons)} memory neurons...")
            saved_count = 0
            for memory in memory_neurons:
                try:
                    if container.graph.neuron_store.create(memory):
                        saved_count += 1
                except Exception as e:
                    print(f"    Error saving {memory.id}: {e}")
            print(f"  Successfully saved {saved_count}/{len(memory_neurons)} memories")
        
        container.shutdown()
        print("[OK] Saved")
        print()
        
        # Reload and verify persistence
        print("=" * 70)
        print("TESTING PERSISTENCE (Reload)")
        print("=" * 70)
        print()
        
        container2 = ApplicationContainer(settings)
        container2.initialize()
        
        try:
            language_model2 = LanguageModel(
                container2.graph,
                container2.compression_engine,
                container2.query_engine,
                container2.training_engine
            )
            
            print("After reload:")
            stats2 = language_model2.get_statistics()
            print(f"  Total neurons: {stats2['total_neurons']:,}")
            print(f"  Memory neurons: {stats2['memory_neurons']:,}")
            print()
            
            # Verify memories still exist
            print("Verifying memories after reload...")
            found_count = 0
            for i, memory_id in enumerate(memory_ids, 1):
                neuron = container2.graph.get_neuron(memory_id)
                if neuron and isinstance(neuron, MemoryNeuron):
                    found_count += 1
                    print(f"  Memory {i}: Found [OK]")
                else:
                    print(f"  Memory {i}: NOT FOUND [FAIL]")
            
            print()
            print(f"Found {found_count}/5 memories after reload")
            
            if found_count == 5:
                print("[OK] All memories persisted correctly!")
            else:
                print("[FAIL] Some memories were lost!")
            
            print()
            
        finally:
            container2.shutdown()
        
        print("=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        container.shutdown()


if __name__ == "__main__":
    test_memory_persistence()
