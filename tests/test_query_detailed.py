"""
Detailed test of the query engine to see what's happening.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
import numpy as np

def main():
    print("=" * 70)
    print("Detailed Query Engine Test")
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
    
    try:
        query_engine = container.query_engine
        compression_engine = container.compression_engine
        graph = container.graph
        
        print(f"Loaded {len(graph.neurons)} neurons")
        print()
        
        # Test query
        query_text = "What is language?"
        print(f"Query: {query_text}")
        print()
        
        # Compress query
        query_vector, meta = compression_engine.compress(query_text)
        print(f"Query vector shape: {query_vector.shape}")
        print(f"Query vector mean: {np.mean(query_vector):.3f}")
        print()
        
        # Get some sample neurons
        sample_neurons = list(graph.neurons.values())[:5]
        
        print("Testing similarity with sample neurons:")
        print("-" * 70)
        for i, neuron in enumerate(sample_neurons):
            if neuron.vector is not None:
                similarity = query_engine._cosine_similarity(query_vector, neuron.vector)
                activation = (similarity + 1.0) / 2.0
                
                content = neuron.source_data[:60] + "..." if len(neuron.source_data) > 60 else neuron.source_data
                print(f"{i+1}. Similarity: {similarity:.3f}, Activation: {activation:.3f}")
                print(f"   Content: {content}")
                print()
        
        # Now test the full query
        print("=" * 70)
        print("Full query test (no propagation):")
        print("-" * 70)
        results = query_engine.query(query_text, top_k=5, propagation_depth=0)
        
        for i, result in enumerate(results):
            content = result.neuron.source_data[:60] + "..." if len(result.neuron.source_data) > 60 else result.neuron.source_data
            similarity_str = f"{result.similarity:.3f}" if result.similarity is not None else "N/A"
            print(f"{i+1}. Activation: {result.activation:.3f}, Similarity: {similarity_str}")
            print(f"   Content: {content}")
            print()
        
    finally:
        container.shutdown()

if __name__ == "__main__":
    main()
