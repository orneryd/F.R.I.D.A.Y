"""
Test continuous improvement over multiple training sessions.
Verifies that the AI gets better with each training round.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel


def test_continuous_improvement():
    print("=" * 70)
    print("TESTING CONTINUOUS IMPROVEMENT")
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
            "How can you help?",
            "What is AI?",
        ]
        
        # Track metrics over sessions
        session_metrics = []
        
        # Run 5 training sessions
        num_sessions = 5
        
        for session in range(1, num_sessions + 1):
            print(f"\n{'=' * 70}")
            print(f"SESSION {session}/{num_sessions}")
            print(f"{'=' * 70}\n")
            
            # Create fresh language model for each session
            language_model = LanguageModel(
                container.graph,
                container.compression_engine,
                container.query_engine,
                container.training_engine,
                enable_self_training=True
            )
            
            # Get initial state
            neurons_before = len(container.graph.neurons)
            
            # Calculate average importance before
            importances_before = [
                n.importance for n in container.graph.neurons.values()
                if hasattr(n, 'importance')
            ]
            avg_importance_before = sum(importances_before) / len(importances_before) if importances_before else 0
            
            # Ask questions
            responses = []
            for question in test_questions:
                response = language_model.generate_response(
                    question,
                    context_size=5,
                    min_activation=0.1,
                    use_reasoning=True
                )
                responses.append(response)
                print(f"Q: {question}")
                print(f"A: {response[:60]}...")
            
            # Consolidate
            print(f"\nConsolidating...")
            if language_model._continuous_learning:
                language_model._continuous_learning.self_training.consolidate_learning()
                
                # Save changes
                print(f"Saving changes...")
                language_model._continuous_learning.self_training.save_learning_state()
            
            # Get final state
            neurons_after = len(container.graph.neurons)
            
            # Calculate average importance after
            importances_after = [
                n.importance for n in container.graph.neurons.values()
                if hasattr(n, 'importance')
            ]
            avg_importance_after = sum(importances_after) / len(importances_after) if importances_after else 0
            
            # Get stats
            if language_model._continuous_learning:
                stats = language_model._continuous_learning.get_statistics()
                
                session_metrics.append({
                    'session': session,
                    'neurons': neurons_after,
                    'neurons_removed': neurons_before - neurons_after,
                    'avg_importance_before': avg_importance_before,
                    'avg_importance_after': avg_importance_after,
                    'importance_change': avg_importance_after - avg_importance_before,
                    'success_rate': stats['success_rate'],
                    'positive': stats['positive_feedback'],
                    'negative': stats['negative_feedback'],
                    'reinforced': stats['neurons_reinforced'],
                    'weakened': stats['neurons_weakened'],
                })
                
                print(f"\nSession {session} Results:")
                print(f"  Neurons: {neurons_before} -> {neurons_after} ({neurons_before - neurons_after:+d})")
                print(f"  Avg importance: {avg_importance_before:.3f} -> {avg_importance_after:.3f} ({avg_importance_after - avg_importance_before:+.3f})")
                print(f"  Success rate: {stats['success_rate']:.1%}")
                print(f"  Feedback: {stats['positive_feedback']}+ / {stats['negative_feedback']}-")
                print(f"  Neurons: {stats['neurons_reinforced']} reinforced, {stats['neurons_weakened']} weakened")
        
        # Analysis
        print(f"\n{'=' * 70}")
        print("CONTINUOUS IMPROVEMENT ANALYSIS")
        print(f"{'=' * 70}\n")
        
        print(f"{'Session':<10} {'Neurons':<10} {'Avg Imp':<12} {'Success':<10} {'Feedback':<12}")
        print("-" * 70)
        
        for m in session_metrics:
            print(
                f"{m['session']:<10} "
                f"{m['neurons']:<10} "
                f"{m['avg_importance_after']:.3f} ({m['importance_change']:+.3f})  "
                f"{m['success_rate']:.1%}{'':>4} "
                f"{m['positive']}+ / {m['negative']}-"
            )
        
        print()
        
        # Check for improvement
        if len(session_metrics) >= 2:
            first = session_metrics[0]
            last = session_metrics[-1]
            
            print("Overall Changes:")
            print(f"  Neurons: {first['neurons']} -> {last['neurons']} ({last['neurons'] - first['neurons']:+d})")
            print(f"  Avg importance: {first['avg_importance_after']:.3f} -> {last['avg_importance_after']:.3f} ({last['avg_importance_after'] - first['avg_importance_after']:+.3f})")
            print(f"  Success rate: {first['success_rate']:.1%} -> {last['success_rate']:.1%}")
            print()
            
            # Check improvements
            improvements = []
            
            if last['avg_importance_after'] > first['avg_importance_after']:
                improvements.append(f"[OK] Average neuron importance increased by {(last['avg_importance_after'] - first['avg_importance_after']):.3f}")
            
            if last['success_rate'] > first['success_rate']:
                improvements.append(f"[OK] Success rate improved by {(last['success_rate'] - first['success_rate']) * 100:.1f} percentage points")
            
            if last['neurons'] < first['neurons']:
                improvements.append(f"[OK] Network pruned by {first['neurons'] - last['neurons']} neurons (removing bad ones)")
            
            # Check if importance is trending up
            importance_trend = [m['avg_importance_after'] for m in session_metrics]
            if importance_trend[-1] > importance_trend[0]:
                improvements.append(f"[OK] Importance trending upward (network getting stronger)")
            
            if improvements:
                print("Improvements Detected:")
                for imp in improvements:
                    print(f"  {imp}")
            else:
                print("[WARN] No clear improvements detected")
            
            print()
            
            # Success criteria
            print("Success Criteria:")
            if last['avg_importance_after'] > first['avg_importance_after']:
                print("  [OK] Network quality improved (higher average importance)")
            else:
                print("  [FAIL] Network quality did not improve")
            
            if last['neurons'] <= first['neurons']:
                print("  [OK] Network optimized (bad neurons removed)")
            else:
                print("  [FAIL] Network not optimized (neurons increased)")
            
            print()
        
    finally:
        container.shutdown()


if __name__ == "__main__":
    test_continuous_improvement()
