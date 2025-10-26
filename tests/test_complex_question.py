"""
Test complex philosophical questions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel

def main():
    print("=" * 70)
    print("Testing Complex Questions")
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
        
        # Test complex questions
        test_questions = [
            "How are you feeling about being an AI?",
            "Do you enjoy helping people?",
            "What do you think about artificial intelligence?",
            "Are you happy?",
            "What's your purpose?",
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"{i}. Q: {question}")
            
            # Show top 3 activated neurons
            results = language_model.understand(question, top_k=3, propagation_depth=0)
            if results:
                print(f"   Top 3 neurons:")
                for j, result in enumerate(results[:3]):
                    content = result.neuron.source_data[:60] + "..." if len(result.neuron.source_data) > 60 else result.neuron.source_data
                    print(f"     {j+1}. [{result.activation:.3f}] {content}")
            
            # Generate response
            response = language_model.generate_response(
                question,
                context_size=5,
                min_activation=0.3,
                propagation_depth=0
            )
            
            print(f"   A: {response}")
            print()
        
        print("=" * 70)
        
    finally:
        container.shutdown()

if __name__ == "__main__":
    main()
