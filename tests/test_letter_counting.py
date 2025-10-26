"""
Test the AI's letter counting abilities with various words.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel

def main():
    print("=" * 70)
    print("Testing AI Letter Counting Abilities")
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
        
        # Test various questions
        test_questions = [
            "How many letter R are in strawberry?",
            "How many letters P in apple?",
            "How many letter O in book?",
            "Count the letter E in tree",
            "How many letter L in hello?",
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"{i}. Q: {question}")
            
            response = language_model.generate_response(
                question,
                context_size=5,
                min_activation=0.5,
                propagation_depth=0
            )
            
            print(f"   A: {response}")
            print()
        
        print("=" * 70)
        
    finally:
        container.shutdown()

if __name__ == "__main__":
    main()
