"""
Test what neurons are activated for language-related queries.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel

def main():
    print("=" * 70)
    print("Testing Language Query")
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
        language_model = LanguageModel(
            container.graph,
            container.compression_engine,
            container.query_engine,
            container.training_engine
        )
        
        # Test query
        query = "What is language?"
        print(f"Query: {query}")
        print()
        
        # Get top 20 results
        results = language_model.understand(query, top_k=20, propagation_depth=0)
        
        print(f"Top 20 activated neurons:")
        print("-" * 70)
        for i, result in enumerate(results):
            content = result.neuron.source_data
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"{i+1:2}. [{result.activation:.3f}] {content}")
        
        print()
        print("=" * 70)
        
        # Check if we have language-related knowledge
        print("\nSearching for language-related neurons in database...")
        language_keywords = ["language", "communication", "speak", "talk", "word", "grammar"]
        
        found_neurons = []
        for neuron in container.graph.neurons.values():
            if hasattr(neuron, 'source_data'):
                content_lower = neuron.source_data.lower()
                for keyword in language_keywords:
                    if keyword in content_lower:
                        found_neurons.append((neuron, keyword))
                        break
        
        print(f"\nFound {len(found_neurons)} neurons with language-related content:")
        for neuron, keyword in found_neurons[:10]:
            content = neuron.source_data
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"  - [{keyword}] {content}")
        
    finally:
        container.shutdown()

if __name__ == "__main__":
    main()
