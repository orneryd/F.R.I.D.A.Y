"""
Test self-training capabilities.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel


def test_self_training():
    print("=" * 70)
    print("TESTING SELF-TRAINING CAPABILITIES")
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
        language_model = LanguageModel(
            container.graph,
            container.compression_engine,
            container.query_engine,
            container.training_engine,
            enable_self_training=True
        )
        
        print("Self-training enabled! The AI will learn from each interaction.")
        print()
        
        # Test questions - ask same questions multiple times
        test_questions = [
            "What is artificial intelligence?",
            "How do you learn?",
            "What do you do when you're not sure?",
            "Why is practice important?",
            "How do you solve a problem?",
        ]
        
        print("Round 1: Initial responses")
        print("-" * 70)
        print()
        
        round1_responses = []
        for i, question in enumerate(test_questions, 1):
            print(f"{i}. Q: {question}")
            response = language_model.generate_response(
                question,
                context_size=5,
                min_activation=0.1,
                use_reasoning=True
            )
            print(f"   A: {response[:100]}...")
            round1_responses.append(response)
            print()
        
        # Get statistics after round 1
        if language_model._continuous_learning:
            stats1 = language_model._continuous_learning.get_statistics()
            print("=" * 70)
            print("SELF-TRAINING STATISTICS (After Round 1)")
            print("=" * 70)
            print(f"Interactions: {stats1['interaction_count']}")
            print(f"Positive feedback: {stats1['positive_feedback']}")
            print(f"Negative feedback: {stats1['negative_feedback']}")
            print(f"Success rate: {stats1['success_rate']:.1%}")
            print(f"Neurons reinforced: {stats1['neurons_reinforced']}")
            print(f"Neurons weakened: {stats1['neurons_weakened']}")
            print(f"New connections: {stats1['new_connections_created']}")
            print()
        
        # Ask same questions again
        print("=" * 70)
        print("Round 2: After self-training")
        print("-" * 70)
        print()
        
        round2_responses = []
        for i, question in enumerate(test_questions, 1):
            print(f"{i}. Q: {question}")
            response = language_model.generate_response(
                question,
                context_size=5,
                min_activation=0.1,
                use_reasoning=True
            )
            print(f"   A: {response[:100]}...")
            round2_responses.append(response)
            
            # Compare with round 1
            if response != round1_responses[i-1]:
                print(f"   ✓ Response improved/changed!")
            else:
                print(f"   - Same response")
            print()
        
        # Final statistics
        if language_model._continuous_learning:
            stats2 = language_model._continuous_learning.get_statistics()
            print("=" * 70)
            print("SELF-TRAINING STATISTICS (After Round 2)")
            print("=" * 70)
            print(f"Total interactions: {stats2['interaction_count']}")
            print(f"Positive feedback: {stats2['positive_feedback']}")
            print(f"Negative feedback: {stats2['negative_feedback']}")
            print(f"Success rate: {stats2['success_rate']:.1%}")
            print(f"Neurons reinforced: {stats2['neurons_reinforced']}")
            print(f"Neurons weakened: {stats2['neurons_weakened']}")
            print(f"New connections: {stats2['new_connections_created']}")
            print(f"Tracked neurons: {stats2['tracked_neurons']}")
            print()
            
            # Save learning state
            print("Saving learning state to database...")
            language_model._continuous_learning.self_training.save_learning_state()
            print("✓ Learning state saved!")
            print()
        
        # AI statistics
        ai_stats = language_model.get_statistics()
        print("=" * 70)
        print("AI STATISTICS")
        print("=" * 70)
        print(f"Total neurons: {ai_stats['total_neurons']}")
        print(f"Total synapses: {ai_stats['total_synapses']}")
        print(f"Connectivity: {ai_stats['average_connectivity']:.2f}")
        print()
        
    finally:
        container.shutdown()


if __name__ == "__main__":
    test_self_training()
