"""
Test that training actually improves the AI over multiple rounds.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel


def test_training_improvement():
    print("=" * 70)
    print("TESTING TRAINING IMPROVEMENT OVER MULTIPLE ROUNDS")
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
        
        # Test questions
        test_questions = [
            "What are you?",
            "Who are you?",
            "What is AI?",
            "Explain machine learning",
            "What is a neural network?",
            "How does learning work?",
            "What is intelligence?",
            "Can you think?",
            "Are you conscious?",
            "Do you have feelings?",
        ]
        
        # Run multiple training rounds
        num_rounds = 3
        round_stats = []
        
        for round_num in range(1, num_rounds + 1):
            print(f"\n{'=' * 70}")
            print(f"ROUND {round_num}/{num_rounds}")
            print(f"{'=' * 70}\n")
            
            # Get initial state
            neurons_before = len(container.graph.neurons)
            synapses_before = len(container.graph.synapses)
            
            # Ask all questions
            for i, question in enumerate(test_questions, 1):
                response = language_model.generate_response(
                    question,
                    context_size=5,
                    min_activation=0.1,
                    use_reasoning=True
                )
                print(f"{i}. Q: {question}")
                print(f"   A: {response[:70]}...")
            
            # Force consolidation
            print(f"\nConsolidating learning...")
            if language_model._continuous_learning:
                language_model._continuous_learning.self_training.consolidate_learning()
            
            # Get stats
            neurons_after = len(container.graph.neurons)
            synapses_after = len(container.graph.synapses)
            
            if language_model._continuous_learning:
                stats = language_model._continuous_learning.get_statistics()
                round_stats.append({
                    'round': round_num,
                    'neurons_before': neurons_before,
                    'neurons_after': neurons_after,
                    'neurons_removed': neurons_before - neurons_after,
                    'synapses_before': synapses_before,
                    'synapses_after': synapses_after,
                    'synapses_removed': synapses_before - synapses_after,
                    'success_rate': stats['success_rate'],
                    'positive': stats['positive_feedback'],
                    'negative': stats['negative_feedback'],
                    'neurons_removed_total': stats['neurons_removed'],
                    'synapses_removed_total': stats['synapses_removed'],
                })
                
                print(f"\nRound {round_num} Results:")
                print(f"  Neurons: {neurons_before} → {neurons_after} (removed: {neurons_before - neurons_after})")
                print(f"  Synapses: {synapses_before} → {synapses_after} (removed: {synapses_before - synapses_after})")
                print(f"  Success rate: {stats['success_rate']:.1%}")
                print(f"  Feedback: {stats['positive_feedback']} positive, {stats['negative_feedback']} negative")
        
        # Summary
        print(f"\n{'=' * 70}")
        print("TRAINING IMPROVEMENT SUMMARY")
        print(f"{'=' * 70}\n")
        
        print(f"{'Round':<8} {'Neurons':<12} {'Synapses':<12} {'Success':<10} {'Feedback':<15}")
        print("-" * 70)
        
        for stats in round_stats:
            neurons_change = f"-{stats['neurons_removed']}" if stats['neurons_removed'] > 0 else "0"
            synapses_change = f"-{stats['synapses_removed']}" if stats['synapses_removed'] > 0 else "0"
            feedback = f"{stats['positive']}+/{stats['negative']}-"
            
            print(f"{stats['round']:<8} {neurons_change:<12} {synapses_change:<12} {stats['success_rate']:.1%}{'':>4} {feedback:<15}")
        
        print()
        
        # Check for improvement
        if len(round_stats) >= 2:
            first_success = round_stats[0]['success_rate']
            last_success = round_stats[-1]['success_rate']
            
            total_neurons_removed = sum(s['neurons_removed'] for s in round_stats)
            total_synapses_removed = sum(s['synapses_removed'] for s in round_stats)
            
            print("Analysis:")
            print(f"  Total neurons removed: {total_neurons_removed}")
            print(f"  Total synapses removed: {total_synapses_removed}")
            print(f"  Success rate change: {first_success:.1%} → {last_success:.1%}")
            
            if last_success > first_success:
                improvement = (last_success - first_success) * 100
                print(f"  ✓ IMPROVEMENT: +{improvement:.1f} percentage points")
            elif last_success == first_success:
                print(f"  = STABLE: No change in success rate")
            else:
                decline = (first_success - last_success) * 100
                print(f"  ⚠ DECLINE: -{decline:.1f} percentage points")
            
            print()
            
            if total_neurons_removed > 0:
                print(f"  ✓ Network is being pruned (removing bad neurons)")
            else:
                print(f"  ⚠ No neurons removed (may need more aggressive training)")
            
            if total_synapses_removed > 0:
                print(f"  ✓ Connections are being optimized (removing bad synapses)")
            else:
                print(f"  ⚠ No synapses removed (may need more aggressive training)")
        
        print()
        
    finally:
        container.shutdown()


if __name__ == "__main__":
    test_training_improvement()
