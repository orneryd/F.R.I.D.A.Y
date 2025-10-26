"""
Test the query engine with the trained AI.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel

def main():
    print("=" * 70)
    print("Testing Query Engine with Trained AI")
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
        
        # Show stats
        stats = language_model.get_statistics()
        print(f"Loaded AI with {stats['total_neurons']} neurons")
        print()
        
        # Test with various questions
        test_questions = [
            "What is language?",
            "Tell me about history",
            "What is a book?",
            "Explain knowledge",
            "What is learning?",
            "Tell me about communication",
        ]
        
        print("Testing query engine with semantic similarity:")
        print("-" * 70)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Q: {question}")
            
            # Test understanding (what neurons are activated)
            results = language_model.understand(question, top_k=5)
            print(f"   Activated {len(results)} neurons")
            
            if results:
                print(f"   Top activations:")
                for j, result in enumerate(results[:3]):
                    content = result.neuron.source_data[:80] + "..." if len(result.neuron.source_data) > 80 else result.neuron.source_data
                    print(f"     {j+1}. [{result.activation:.3f}] {content}")
            
            # Generate response
            response = language_model.generate_response(
                question,
                context_size=3,
                min_activation=0.1
            )
            print(f"   A: {response[:150]}{'...' if len(response) > 150 else ''}")
        
        print()
        print("=" * 70)
        print("Test Complete!")
        print("=" * 70)
        
    finally:
        container.shutdown()

if __name__ == "__main__":
    main()
