"""
Test the AI's conversation abilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel

def main():
    print("=" * 70)
    print("Testing AI Conversation Abilities")
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
        
        # Test conversational questions
        test_questions = [
            "How are you?",
            "What are you?",
            "Can you help me?",
            "Thank you",
            "What can you do?",
            "Who are you?",
            "Hello",
            "How's it going?",
        ]
        
        print("Testing conversational responses:")
        print("-" * 70)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Q: {question}")
            
            response = language_model.generate_response(
                question,
                context_size=5,
                min_activation=0.3,
                propagation_depth=0
            )
            
            print(f"   A: {response}")
        
        print()
        print("=" * 70)
        print("Conversation Test Complete!")
        print("=" * 70)
        
    finally:
        container.shutdown()

if __name__ == "__main__":
    main()
