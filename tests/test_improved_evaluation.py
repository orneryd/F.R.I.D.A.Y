"""
Test improved auto-evaluation system.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel


def test_improved_evaluation():
    print("=" * 70)
    print("TESTING IMPROVED AUTO-EVALUATION")
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
        
        # Test questions with expected good answers
        test_cases = [
            {
                'question': 'What are you?',
                'expected': 'Should recognize AI/assistant answer as GOOD'
            },
            {
                'question': 'Who are you?',
                'expected': 'Should recognize identity answer as GOOD'
            },
            {
                'question': 'What is AI?',
                'expected': 'Should recognize definition as GOOD'
            },
            {
                'question': 'Can you help me?',
                'expected': 'Should recognize helpful response as GOOD'
            },
            {
                'question': 'What is machine learning?',
                'expected': 'Should recognize explanation as GOOD'
            },
        ]
        
        print("Testing responses and evaluation...")
        print("-" * 70)
        
        results = []
        for i, test in enumerate(test_cases, 1):
            question = test['question']
            expected = test['expected']
            
            print(f"\n{i}. Q: {question}")
            print(f"   Expected: {expected}")
            
            # Generate response
            response = language_model.generate_response(
                question,
                context_size=5,
                min_activation=0.1,
                use_reasoning=True
            )
            
            print(f"   A: {response[:80]}...")
            
            # Check what evaluation it got
            if language_model._continuous_learning:
                stats = language_model._continuous_learning.get_statistics()
                total = stats['interaction_count']
                positive = stats['positive_feedback']
                negative = stats['negative_feedback']
                
                # Determine if this response was positive or negative
                if i == 1:
                    was_positive = positive > 0
                else:
                    was_positive = positive > results[-1]['positive']
                
                results.append({
                    'question': question,
                    'response': response,
                    'positive': positive,
                    'negative': negative,
                    'was_positive': was_positive
                })
                
                if was_positive:
                    print(f"   ✓ Evaluated as POSITIVE")
                else:
                    print(f"   ✗ Evaluated as NEGATIVE")
        
        # Summary
        print()
        print("=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        if language_model._continuous_learning:
            stats = language_model._continuous_learning.get_statistics()
            
            print(f"\nTotal interactions:  {stats['interaction_count']}")
            print(f"Positive feedback:   {stats['positive_feedback']}")
            print(f"Negative feedback:   {stats['negative_feedback']}")
            print(f"Success rate:        {stats['success_rate']:.1%}")
            print()
            
            # Analyze results
            positive_count = sum(1 for r in results if r['was_positive'])
            negative_count = len(results) - positive_count
            
            print(f"Good answers recognized: {positive_count}/{len(results)}")
            print(f"Bad evaluations:         {negative_count}/{len(results)}")
            print()
            
            if stats['success_rate'] >= 0.6:
                print("✓ GOOD: Evaluation is working well (≥60% positive)")
            elif stats['success_rate'] >= 0.4:
                print("⚠ OK: Evaluation is acceptable (40-60% positive)")
            else:
                print("✗ BAD: Evaluation is too strict (<40% positive)")
            
            print()
            
            # Show which responses were evaluated incorrectly
            print("Response Analysis:")
            print("-" * 70)
            for i, result in enumerate(results, 1):
                status = "✓ POSITIVE" if result['was_positive'] else "✗ NEGATIVE"
                print(f"{i}. {result['question']}")
                print(f"   {status}")
                print(f"   Response: {result['response'][:60]}...")
                print()
        
    finally:
        container.shutdown()


if __name__ == "__main__":
    test_improved_evaluation()
