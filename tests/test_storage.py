"""
Test if storage is properly attached to the graph.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings

def main():
    print("Testing storage attachment...")
    
    db_path = "test_storage.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    settings = Settings(database_path=db_path)
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        graph = container.graph
        
        print(f"Graph has neuron_store: {hasattr(graph, 'neuron_store')}")
        print(f"Graph has synapse_store: {hasattr(graph, 'synapse_store')}")
        
        if hasattr(graph, 'neuron_store'):
            print(f"neuron_store is None: {graph.neuron_store is None}")
        
        if hasattr(graph, 'synapse_store'):
            print(f"synapse_store is None: {graph.synapse_store is None}")
        
        # Try to add a neuron
        from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
        from neuron_system.core.vector3d import Vector3D
        
        neuron = KnowledgeNeuron(
            position=Vector3D(0, 0, 0),
            content="Test content"
        )
        graph.add_neuron(neuron)
        
        print(f"\nAdded neuron: {neuron.id}")
        print(f"Neurons in graph: {len(graph.neurons)}")
        
        # Check database
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM neurons;")
        count = cursor.fetchone()[0]
        print(f"Neurons in database: {count}")
        conn.close()
        
    finally:
        container.shutdown()
    
    # Check again after shutdown
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM neurons;")
    count = cursor.fetchone()[0]
    print(f"\nAfter shutdown - Neurons in database: {count}")
    conn.close()

if __name__ == "__main__":
    main()
