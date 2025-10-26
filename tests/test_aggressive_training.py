"""
Test aggressive training system - verify neurons and synapses are actually removed.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel


def test_aggressive_training():
    print("=" * 70)
    print("TESTING AGGRESSIVE TRAINING SYSTEM")
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
        
        # Get initial counts
        initial_neurons = len(container.graph.neurons)
        initial_synapses = len(container.graph.synapses)
        
        print(f"Initial state:")
        print(f"  Neurons: {initial_neurons}")
        print(f"  Synapses: {initial_synapses}")
        print()
        
        # Test with questions that should produce bad responses
        print("Testing with various questions...")
        print("-" * 70)
        
        test_questions = [
            "What are you?",
            "Who are you?",
            "What is your name?",
            "Tell me about yourself",
            "What can you do?",
            "How can you help me?",
            "What is AI?",
            "Explain machine learning",
            "What is a neural network?",
            "How does learning work?",
            "What is intelligence?",
            "Can you think?",
            "Are you conscious?",
            "What is consciousness?",
            "Do you have feelings?",
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Q: {question}")
            response = language_model.generate_response(
                question,
                context_size=5,
                min_activation=0.1,
                use_reasoning=True
            )
            print(f"   A: {response[:80]}...")
        
        print()
        print("=" * 70)
        print("FORCING CONSOLIDATION")
        print("=" * 70)
        
        # Force consolidation to trigger removal
        if language_model._continuous_learning:
            language_model._continuous_learning.self_training.consolidate_learning()
        
        # Get final counts
        final_neurons = len(container.graph.neurons)
        final_synapses = len(container.graph.synapses)
        
        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Initial neurons:  {initial_neurons}")
        print(f"Final neurons:    {final_neurons}")
        print(f"Neurons removed:  {initial_neurons - final_neurons}")
        print()
        print(f"Initial synapses: {initial_synapses}")
        print(f"Final synapses:   {final_synapses}")
        print(f"Synapses removed: {initial_synapses - final_synapses}")
        print()
        
        # Get detailed statistics
        if language_model._continuous_learning:
            stats = language_model._continuous_learning.get_statistics()
            print("=" * 70)
            print("TRAINING STATISTICS")
            print("=" * 70)
            print(f"Total interactions:        {stats['interaction_count']}")
            print(f"Positive feedback:         {stats['positive_feedback']}")
            print(f"Negative feedback:         {stats['negative_feedback']}")
            print(f"Success rate:              {stats['success_rate']:.1%}")
            print()
            print(f"Neurons reinforced:        {stats['neurons_reinforced']}")
            print(f"Neurons weakened:          {stats['neurons_weakened']}")
            print(f"Neurons removed:           {stats['neurons_removed']}")
            print(f"Net neuron change:         {stats['net_neuron_change']}")
            print()
            print(f"Synapses created:          {stats['new_connections_created']}")
            print(f"Synapses removed:          {stats['synapses_removed']}")
            print()
            print(f"Generic responses:         {stats['generic_responses_detected']}")
            print(f"Diversity violations:      {stats['diversity_violations']}")
            print(f"Net quality:               {stats['net_quality']}")
            print()
        
        # Verify that training actually happened
        print("=" * 70)
        print("VERIFICATION")
        print("=" * 70)
        
        if initial_neurons == final_neurons and initial_synapses == final_synapses:
            print("❌ WARNING: No neurons or synapses were removed!")
            print("   Training may not be working correctly.")
        else:
            print("✓ Training is working - network structure changed")
            
        if stats['neurons_removed'] == 0:
            print("❌ WARNING: No neurons were removed by training system!")
        else:
            print(f"✓ Training removed {stats['neurons_removed']} bad neurons")
            
        if stats['synapses_removed'] == 0:
            print("❌ WARNING: No synapses were removed by training system!")
        else:
            print(f"✓ Training removed {stats['synapses_removed']} bad synapses")
        
        if stats['negative_feedback'] == 0:
            print("❌ WARNING: No negative feedback detected!")
            print("   Auto-evaluation may be too lenient.")
        else:
            print(f"✓ Auto-evaluation detected {stats['negative_feedback']} bad responses")
        
        print()
        
    finally:
        container.shutdown()


if __name__ == "__main__":
    test_aggressive_training()
