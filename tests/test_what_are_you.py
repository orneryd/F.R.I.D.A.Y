"""
Detailed test for "What are you?" question.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel

def main():
    print("=" * 70)
    print("Detailed Test: What are you?")
    print("=" * 70)
    print()
    
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
        
        question = "What are you?"
        print(f"Q: {question}")
        print()
        
        # Show top 10 activated neurons
        results = language_model.understand(question, top_k=10, propagation_depth=0)
        if results:
            print("Top 10 activated neurons:")
            for i, result in enumerate(results[:10]):
                content = result.neuron.source_data[:80] + "..." if len(result.neuron.source_data) > 80 else result.neuron.source_data
                print(f"  {i+1:2}. [{result.activation:.3f}] {content}")
        print()
        
        # Generate response
        response = language_model.generate_response(
            question,
            context_size=10,
            min_activation=0.3,
            propagation_depth=0
        )
        
        print(f"A: {response}")
        
    finally:
        container.shutdown()

if __name__ == "__main__":
    main()
