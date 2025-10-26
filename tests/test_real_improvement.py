"""
Test REAL improvement - verify that training actually makes responses better.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel


def test_real_improvement():
    print("=" * 70)
    print("TESTING REAL IMPROVEMENT")
    print("=" * 70)
    print()
    
    settings = Settings(
        database_path="comprehensive_ai.db",
        spatial_bounds_min=(-500.0, -500.0, -500.0),
        spatial_bounds_max=(500.0, 500.0, 500.0)
    )
    
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        # Test questions
        test_questions = [
            "What are you?",
            "Who are you?",
            "What can you do?",
            "What is AI?",
            "Explain machine learning",
        ]
        
        # Track responses over rounds
        round_responses = []
        
        # Run 3 training rounds
        for round_num in range(1, 4):
            print(f"\n{'=' * 70}")
            print(f"ROUND {round_num}/3")
            print(f"{'=' * 70}\n")
            
            # Create language model
            language_model = LanguageModel(
                container.graph,
                container.compression_engine,
                container.query_engine,
                container.training_engine,
                enable_self_training=True
            )
            
            # Get responses
            responses = {}
            for question in test_questions:
                response = language_model.generate_response(
                    question,
                    context_size=5,
                    min_activation=0.1,
                    use_reasoning=True
                )
                responses[question] = response
                print(f"Q: {question}")
                print(f"A: {response[:70]}...")
                print()
            
            round_responses.append(responses)
            
            # Consolidate
            if language_model._continuous_learning:
                language_model._continuous_learning.self_training.consolidate_learning()
                language_model._continuous_learning.self_training.save_learning_state()
                
                stats = language_model._continuous_learning.get_statistics()
                print(f"Round {round_num} Stats:")
                print(f"  Success rate: {stats['success_rate']:.1%}")
                print(f"  Positive: {stats['positive_feedback']}, Negative: {stats['negative_feedback']}")
                print(f"  Neurons removed: {stats['neurons_removed']}")
        
        # Analysis
        print(f"\n{'=' * 70}")
        print("IMPROVEMENT ANALYSIS")
        print(f"{'=' * 70}\n")
        
        for question in test_questions:
            print(f"Question: {question}")
            print("-" * 70)
            
            for i, responses in enumerate(round_responses, 1):
                response = responses[question]
                print(f"Round {i}: {response[:60]}...")
            
            # Check if responses improved
            first_response = round_responses[0][question]
            last_response = round_responses[-1][question]
            
            if "I'm not confident" in first_response and "I'm not confident" not in last_response:
                print("[OK] Improved: No longer uncertain")
            elif len(last_response) > len(first_response) * 1.2:
                print("[OK] Improved: More detailed response")
            elif first_response == last_response:
                print("[SAME] No change")
            else:
                print("[CHANGED] Response changed")
            
            print()
        
    finally:
        container.shutdown()


if __name__ == "__main__":
    test_real_improvement()
