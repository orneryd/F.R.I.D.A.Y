"""
Example: Using the main entry point programmatically

This example demonstrates how to use the ApplicationContainer
to initialize and use the neuron system programmatically.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
from neuron_system.core.vector3d import Vector3D
from neuron_system.core.synapse import Synapse


def main():
    """Main example function."""
    print("=" * 70)
    print("Example: Using the Main Entry Point")
    print("=" * 70)
    print()
    
    # Create custom settings
    settings = Settings(
        database_path="example_main.db",
        spatial_bounds_min=(-50.0, -50.0, -50.0),
        spatial_bounds_max=(50.0, 50.0, 50.0),
        default_top_k=5
    )
    
    # Initialize application container
    print("Initializing application container...")
    container = ApplicationContainer(settings)
    container.initialize()
    print()
    
    try:
        # Get components
        graph = container.graph
        compression_engine = container.compression_engine
        query_engine = container.query_engine
        training_engine = container.training_engine
        
        print(f"System initialized with {len(graph.neurons)} neurons")
        print()
        
        # Example 1: Create knowledge neurons
        print("Example 1: Creating knowledge neurons")
        print("-" * 70)
        
        knowledge_items = [
            "Python is a high-level programming language",
            "JavaScript is used for web development",
            "Machine learning is a subset of artificial intelligence",
            "Neural networks are inspired by biological neurons",
            "Deep learning uses multiple layers of neural networks"
        ]
        
        neurons = []
        for i, text in enumerate(knowledge_items):
            # Compress text to vector
            vector = compression_engine.compress(text)
            
            # Create neuron
            neuron = KnowledgeNeuron(
                position=Vector3D(i * 10, 0, 0),
                vector=vector,
                source_data=text,
                compression_ratio=len(text) / 384,
                semantic_tags=["knowledge", "example"]
            )
            
            # Add to graph
            graph.add_neuron(neuron)
            neurons.append(neuron)
            print(f"  Created neuron {i+1}: {text[:50]}...")
        
        print(f"\nTotal neurons: {len(graph.neurons)}")
        print()
        
        # Example 2: Create synapses between related neurons
        print("Example 2: Creating synapses")
        print("-" * 70)
        
        # Connect Python to programming concepts
        synapse1 = Synapse(
            source_neuron_id=neurons[0].id,
            target_neuron_id=neurons[2].id,
            weight=0.7
        )
        graph.add_synapse(synapse1)
        print(f"  Connected: Python -> Machine Learning (weight: 0.7)")
        
        # Connect ML to neural networks
        synapse2 = Synapse(
            source_neuron_id=neurons[2].id,
            target_neuron_id=neurons[3].id,
            weight=0.9
        )
        graph.add_synapse(synapse2)
        print(f"  Connected: ML -> Neural Networks (weight: 0.9)")
        
        # Connect neural networks to deep learning
        synapse3 = Synapse(
            source_neuron_id=neurons[3].id,
            target_neuron_id=neurons[4].id,
            weight=0.8
        )
        graph.add_synapse(synapse3)
        print(f"  Connected: Neural Networks -> Deep Learning (weight: 0.8)")
        
        print(f"\nTotal synapses: {len(graph.synapses)}")
        print()
        
        # Example 3: Query the system
        print("Example 3: Querying the system")
        print("-" * 70)
        
        query_text = "artificial intelligence and neural networks"
        print(f"Query: '{query_text}'")
        print()
        
        results = query_engine.query(query_text, top_k=3, propagation_depth=2)
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            neuron = result.neuron
            if hasattr(neuron, 'source_data'):
                print(f"  {i}. Activation: {result.activation:.4f}")
                print(f"     Content: {neuron.source_data}")
                print()
        
        # Example 4: Training - adjust a neuron
        print("Example 4: Training - adjusting neuron vectors")
        print("-" * 70)
        
        # Get the first neuron
        target_neuron = neurons[0]
        print(f"Adjusting neuron: {target_neuron.source_data[:50]}...")
        
        # Create a target vector (e.g., move it closer to ML concepts)
        target_vector = compression_engine.compress(
            "Python is used for machine learning and AI"
        )
        
        # Adjust the neuron
        training_engine.adjust_neuron(
            target_neuron.id,
            target_vector,
            learning_rate=0.3
        )
        
        print("  Neuron vector adjusted successfully")
        print()
        
        # Example 5: Strengthen a synapse
        print("Example 5: Training - strengthening synapses")
        print("-" * 70)
        
        print(f"Original synapse weight: {synapse1.weight}")
        training_engine.strengthen_synapse(synapse1.id, delta=0.1)
        print(f"New synapse weight: {synapse1.weight}")
        print()
        
        # Example 6: System statistics
        print("Example 6: System statistics")
        print("-" * 70)
        print(f"Total neurons: {len(graph.neurons)}")
        print(f"Total synapses: {len(graph.synapses)}")
        print(f"Spatial bounds: {settings.spatial_bounds_min} to {settings.spatial_bounds_max}")
        print(f"Database: {settings.database_path}")
        print()
        
    finally:
        # Cleanup
        print("Shutting down...")
        container.shutdown()
        print("Done!")
        print()
        
        # Clean up example database
        import os
        if os.path.exists("example_main.db"):
            os.remove("example_main.db")
            print("Cleaned up example database")


if __name__ == "__main__":
    main()
