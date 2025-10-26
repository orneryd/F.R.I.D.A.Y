"""
Example demonstrating storage layer usage.

Shows how to persist neurons and synapses, perform incremental saves,
create backups, and restore data.
"""

import numpy as np
from uuid import uuid4
from datetime import datetime

from neuron_system.core.vector3d import Vector3D
from neuron_system.core.synapse import Synapse, SynapseType
from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
from neuron_system.neuron_types.tool_neuron import ToolNeuron
from neuron_system.storage import (
    get_database_manager,
    NeuronStore,
    SynapseStore,
    SerializationManager,
    close_database,
)


def main():
    """Demonstrate storage layer functionality."""
    
    print("=" * 60)
    print("Storage Layer Usage Example")
    print("=" * 60)
    
    # Initialize storage layer
    db_manager = get_database_manager("example_neuron_system.db")
    neuron_store = NeuronStore(db_manager)
    synapse_store = SynapseStore(db_manager)
    serialization_manager = SerializationManager(db_manager)
    
    print("\n1. Creating neurons...")
    
    # Create knowledge neurons
    neurons = {}
    for i in range(5):
        neuron = KnowledgeNeuron(
            source_data=f"Knowledge about topic {i}",
            compression_ratio=100.0,
            semantic_tags=[f"topic_{i}", "example"]
        )
        neuron.id = uuid4()
        neuron.position = Vector3D(float(i * 10), float(i * 10), float(i * 10))
        neuron.vector = np.random.rand(384).astype(np.float32)
        neuron.created_at = datetime.now()
        neuron.modified_at = datetime.now()
        
        neurons[neuron.id] = neuron
        neuron_store.create(neuron)
    
    print(f"   Created {len(neurons)} knowledge neurons")
    
    # Create a tool neuron
    tool = ToolNeuron(
        function_signature="calculate(x: float, y: float) -> float",
        executable_code="return x + y",
        input_schema={"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}},
        output_schema={"type": "number"}
    )
    tool.id = uuid4()
    tool.position = Vector3D(50.0, 50.0, 50.0)
    tool.vector = np.random.rand(384).astype(np.float32)
    tool.created_at = datetime.now()
    tool.modified_at = datetime.now()
    
    neurons[tool.id] = tool
    neuron_store.create(tool)
    print(f"   Created 1 tool neuron")
    
    print("\n2. Creating synapses...")
    
    # Create synapses between neurons
    synapses = {}
    neuron_list = list(neurons.values())
    for i in range(len(neuron_list) - 1):
        synapse = Synapse(
            id=uuid4(),
            source_neuron_id=neuron_list[i].id,
            target_neuron_id=neuron_list[i + 1].id,
            weight=0.5 + (i * 0.1),
            synapse_type=SynapseType.KNOWLEDGE
        )
        synapses[synapse.id] = synapse
        synapse_store.create(synapse)
    
    print(f"   Created {len(synapses)} synapses")
    
    print("\n3. Querying data...")
    
    # Query neurons by type
    knowledge_neurons = neuron_store.list_by_type("knowledge")
    print(f"   Found {len(knowledge_neurons)} knowledge neurons")
    
    tool_neurons = neuron_store.list_by_type("tool")
    print(f"   Found {len(tool_neurons)} tool neurons")
    
    # Query synapses
    first_neuron_id = neuron_list[0].id
    outgoing_synapses = synapse_store.list_by_source(first_neuron_id)
    print(f"   Neuron {first_neuron_id} has {len(outgoing_synapses)} outgoing synapses")
    
    print("\n4. Updating data...")
    
    # Modify a neuron
    first_neuron = neurons[first_neuron_id]
    first_neuron.source_data = "Updated knowledge content"
    first_neuron.modified_at = datetime.now()
    neuron_store.update(first_neuron)
    
    # Track the change
    serialization_manager.change_tracker.mark_neuron_modified(first_neuron_id)
    print(f"   Updated neuron {first_neuron_id}")
    
    # Modify a synapse
    first_synapse = list(synapses.values())[0]
    first_synapse.strengthen(0.1)
    synapse_store.update(first_synapse)
    
    serialization_manager.change_tracker.mark_synapse_modified(first_synapse.id)
    print(f"   Updated synapse {first_synapse.id}")
    
    print("\n5. Incremental save...")
    
    # Save only modified data
    save_stats = serialization_manager.save_incremental(neurons, synapses)
    print(f"   Saved {save_stats['neurons_saved']} neurons and {save_stats['synapses_saved']} synapses")
    print(f"   Duration: {save_stats['duration_seconds']:.4f} seconds")
    
    print("\n6. Creating backup...")
    
    # Create a backup
    backup_path = serialization_manager.create_backup()
    print(f"   Backup created: {backup_path}")
    
    # List all backups
    backups = serialization_manager.list_backups()
    print(f"   Total backups: {len(backups)}")
    
    print("\n7. Data integrity check...")
    
    # Verify data integrity
    integrity = serialization_manager.verify_integrity()
    if integrity["valid"]:
        print("   ✓ Data integrity verified")
    else:
        print(f"   ✗ Issues found: {integrity['issues']}")
    
    print("\n8. Export to JSON...")
    
    # Export entire database to JSON
    export_stats = serialization_manager.export_to_json("neuron_system_export.json")
    print(f"   Exported {export_stats['neurons_exported']} neurons and {export_stats['synapses_exported']} synapses")
    print(f"   File size: {export_stats['file_size_bytes']} bytes")
    
    print("\n9. Database statistics...")
    
    # Get database stats
    stats = db_manager.get_stats()
    print(f"   Total neurons: {stats['neuron_count']}")
    print(f"   Total synapses: {stats['synapse_count']}")
    print(f"   Database size: {stats['database_size_bytes']} bytes")
    
    print("\n10. Batch operations...")
    
    # Create multiple neurons in one transaction
    batch_neurons = []
    for i in range(10):
        neuron = KnowledgeNeuron(
            source_data=f"Batch knowledge {i}",
            compression_ratio=75.0,
            semantic_tags=["batch"]
        )
        neuron.id = uuid4()
        neuron.position = Vector3D(float(i), float(i), float(i))
        neuron.vector = np.random.rand(384).astype(np.float32)
        neuron.created_at = datetime.now()
        neuron.modified_at = datetime.now()
        batch_neurons.append(neuron)
    
    count = neuron_store.batch_create(batch_neurons)
    print(f"   Batch created {count} neurons")
    
    # Update final stats
    final_stats = db_manager.get_stats()
    print(f"\n   Final count: {final_stats['neuron_count']} neurons, {final_stats['synapse_count']} synapses")
    
    # Clean up
    close_database()
    
    print("\n" + "=" * 60)
    print("✅ Storage layer example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
