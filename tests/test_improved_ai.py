"""
Test the improved AI with more complex questions.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel


def test_ai():
    print("=" * 70)
    print("TESTING IMPROVED AI")
    print("=" * 70)
    print()
    
    # Load the trained AI
    settings = Settings(
        database_path="comprehensive_ai.db",
        spatial_bounds_min=(-500.0, -500.0, -500.0),
        spatial_bounds_max=(500.0, 500.0, 500.0)
    )
    
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        # Create language model
        language_model = LanguageModel(
            container.graph,
            container.compression_engine,
            container.query_engine,
            container.training_engine
        )
        
        # Test questions - more complex and conversational
        test_questions = [
            # Identity questions
            "What are you?",
            "Who are you?",
            "What kind of AI are you?",
            
            # Capability questions
            "Can you help me?",
            "What can you do?",
            "Are you smart?",
            
            # Conversational
            "How are you?",
            "Hello",
            "Thank you",
            
            # Complex questions
            "Do you have feelings?",
            "Can you learn?",
            "Are you real?",
            
            # Problem-solving
            "I need help",
            "I'm confused",
            "Can you teach me?",
            
            # Technology
            "What is AI?",
            "How does AI work?",
            "What is machine learning?",
            
            # Meta questions
            "Why are you helping me?",
            "Are you always right?",
            "Can I trust you?",
        ]
        
        print("Testing conversational abilities:")
        print("-" * 70)
        print()
        
        for i, question in enumerate(test_questions, 1):
            print(f"{i}. Q: {question}")
            response = language_model.generate_response(
                question,
                context_size=5,
                min_activation=0.1,
                propagation_depth=0
            )
            print(f"   A: {response}")
            print()
        
        # Statistics
        stats = language_model.get_statistics()
        print("=" * 70)
        print("AI STATISTICS")
        print("=" * 70)
        print(f"Total neurons: {stats['total_neurons']}")
        print(f"Knowledge neurons: {stats['knowledge_neurons']}")
        print(f"Total synapses: {stats['total_synapses']}")
        print(f"Average connectivity: {stats['average_connectivity']:.2f}")
        print()
        
    finally:
        container.shutdown()


if __name__ == "__main__":
    test_ai()
