"""
Debug script to analyze the vectors in the trained AI.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
import numpy as np

def main():
    print("=" * 70)
    print("Debugging Vectors in Trained AI")
    print("=" * 70)
    print()
    
    # Load the trained AI
    db_path = "comprehensive_ai.db"
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found: {db_path}")
        return
    
    settings = Settings(database_path=db_path)
    container = ApplicationContainer(settings)
    container.initialize()
    
    print(f"Loaded {len(container.graph.neurons)} neurons from database")
    print()
    
    try:
        graph = container.graph
        compression_engine = container.compression_engine
        
        print(f"Total neurons: {len(graph.neurons)}")
        
        # Analyze vectors
        neurons_with_vectors = []
        for neuron in graph.neurons.values():
            if hasattr(neuron, 'vector') and neuron.vector is not None:
                neurons_with_vectors.append(neuron)
        
        print(f"Neurons with vectors: {len(neurons_with_vectors)}")
        print()
        
        if len(neurons_with_vectors) > 0:
            # Sample a few neurons
            sample_neurons = neurons_with_vectors[:5]
            
            print("Sample neuron analysis:")
            print("-" * 50)
            
            for i, neuron in enumerate(sample_neurons):
                print(f"\nNeuron {i+1}:")
                print(f"  Type: {type(neuron.vector)}")
                print(f"  Shape: {np.asarray(neuron.vector).shape}")
                print(f"  Length: {len(np.asarray(neuron.vector).flatten())}")
                print(f"  Min/Max: {np.min(neuron.vector):.3f} / {np.max(neuron.vector):.3f}")
                print(f"  Mean: {np.mean(neuron.vector):.3f}")
                print(f"  Std: {np.std(neuron.vector):.3f}")
                
                if hasattr(neuron, 'source_data'):
                    content = neuron.source_data[:100] + "..." if len(neuron.source_data) > 100 else neuron.source_data
                    print(f"  Content: {content}")
            
            # Test similarity between different texts
            print("\n" + "=" * 50)
            print("Testing compression engine:")
            print("-" * 50)
            
            test_texts = [
                "What is language?",
                "Tell me about history",
                "Language is communication",
                "History is the past",
                "Books contain knowledge"
            ]
            
            vectors = []
            for text in test_texts:
                vector, metadata = compression_engine.compress(text)
                vectors.append(vector)
                print(f"\nText: {text}")
                print(f"  Vector shape: {vector.shape}")
                print(f"  Vector mean: {np.mean(vector):.3f}")
                print(f"  Vector std: {np.std(vector):.3f}")
            
            # Calculate similarities between test vectors
            print("\n" + "=" * 50)
            print("Similarity matrix:")
            print("-" * 50)
            
            from neuron_system.engines.query import QueryEngine
            query_engine = QueryEngine(graph, compression_engine)
            
            print("\n     ", end="")
            for i, text in enumerate(test_texts):
                print(f"{i:6}", end="")
            print()
            
            for i, vec1 in enumerate(vectors):
                print(f"{i:2}: ", end="")
                for j, vec2 in enumerate(vectors):
                    sim = query_engine._cosine_similarity(vec1, vec2)
                    print(f"{sim:6.3f}", end="")
                print(f"  {test_texts[i][:20]}...")
        
    finally:
        container.shutdown()

if __name__ == "__main__":
    main()
