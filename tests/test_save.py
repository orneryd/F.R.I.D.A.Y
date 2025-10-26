"""
Test saving neurons to database.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
from neuron_system.core.vector3d import Vector3D
import sqlite3

def main():
    print("Testing manual save...")
    
    db_path = "test_save.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    settings = Settings(database_path=db_path)
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        graph = container.graph
        
        # Add a neuron
        neuron = KnowledgeNeuron(
            position=Vector3D(0, 0, 0),
            content="Test content"
        )
        graph.add_neuron(neuron)
        
        print(f"Added neuron: {neuron.id}")
        print(f"Neurons in graph: {len(graph.neurons)}")
        print(f"Neuron has position: {neuron.position is not None}")
        
        # Try to save manually
        print("\nTrying manual save...")
        neurons_to_save = [n for n in graph.neurons.values() 
                         if hasattr(n, 'position') and n.position is not None]
        print(f"Neurons to save: {len(neurons_to_save)}")
        
        if neurons_to_save and graph.neuron_store:
            print("Calling batch_create...")
            count = graph.neuron_store.batch_create(neurons_to_save)
            print(f"batch_create completed: {count} neurons created")
        
        # Check database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM neurons;")
        count = cursor.fetchone()[0]
        print(f"Neurons in database: {count}")
        
        if count > 0:
            cursor.execute("SELECT id, neuron_type FROM neurons LIMIT 1;")
            row = cursor.fetchone()
            print(f"Sample neuron: ID={row[0]}, Type={row[1]}")
        
        conn.close()
        
    finally:
        container.shutdown()

if __name__ == "__main__":
    main()
