"""
Test self-reflection capabilities.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel


def test_self_reflection():
    print("=" * 70)
    print("TESTING SELF-REFLECTION CAPABILITIES")
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
            container.training_engine
        )
        
        # Test questions that might produce bad responses
        test_cases = [
            ("What are you?", "Should give clear answer"),
            ("How do you learn?", "Should explain learning process"),
            ("What do you do when you're not sure?", "Should explain uncertainty handling"),
            ("How do you know if you're wrong?", "Should explain self-validation"),
            ("What is the difference between knowledge and wisdom?", "Should explain comparison"),
            ("Why is practice important?", "Should explain causation"),
            ("How do you solve a problem?", "Should give step-by-step"),
            ("What is artificial intelligence?", "Should give detailed explanation"),
            ("Tell me everything", "Should handle ambiguity"),
            ("Is it good?", "Should ask for clarification"),
        ]
        
        print("Testing self-reflection with various questions:")
        print("-" * 70)
        print()
        
        for i, (question, expected) in enumerate(test_cases, 1):
            print(f"{i}. Q: {question}")
            print(f"   Expected: {expected}")
            
            response = language_model.generate_response(
                question,
                context_size=7,
                min_activation=0.1,
                use_reasoning=True
            )
            
            print(f"   A: {response}")
            
            # Check response quality
            if len(response) < 20:
                print(f"   ⚠ Response might be too short")
            elif "= =" in response:
                print(f"   ⚠ Response contains artifacts")
            elif "When asked" in response or "When someone" in response:
                print(f"   ⚠ Response contains meta-instructions")
            else:
                print(f"   ✓ Response looks good")
            
            print()
        
        # Get self-reflection statistics
        if hasattr(language_model, '_self_reflection'):
            stats = language_model._self_reflection.get_statistics()
            print("=" * 70)
            print("SELF-REFLECTION STATISTICS")
            print("=" * 70)
            print(f"Total reflections: {stats['total_reflections']}")
            print(f"Total corrections: {stats['total_corrections']}")
            print(f"Correction rate: {stats['correction_rate']:.1%}")
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
    test_self_reflection()
