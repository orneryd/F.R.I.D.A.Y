"""
Test reasoning capabilities of the AI.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel


def test_reasoning():
    print("=" * 70)
    print("TESTING AI REASONING CAPABILITIES")
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
        
        # Test questions requiring reasoning
        test_questions = [
            # Mathematical reasoning
            ("What is 2 plus 2?", "Should use math reasoning"),
            ("How do you add numbers?", "Should explain addition"),
            
            # Logical reasoning
            ("Why is the sky blue?", "Should use causal reasoning"),
            ("How does reasoning work?", "Should explain reasoning"),
            ("What is logic?", "Should define logic"),
            
            # Counting and analysis
            ("How do you count letters?", "Should explain counting method"),
            ("What does counting mean?", "Should define counting"),
            
            # Problem-solving
            ("How do you solve a problem?", "Should explain problem-solving"),
            ("What is a pattern?", "Should explain patterns"),
            
            # Comparison
            ("What is the difference between A and B?", "Should explain comparison"),
            ("How do you compare things?", "Should explain comparison method"),
            
            # Cause and effect
            ("What causes rain?", "Should explain causation"),
            ("Why do things happen?", "Should explain cause-effect"),
            
            # Abstract reasoning
            ("What is an abstract concept?", "Should explain abstraction"),
            ("How do you think abstractly?", "Should explain abstract thinking"),
        ]
        
        print("Testing reasoning with various question types:")
        print("-" * 70)
        print()
        
        for i, (question, expected) in enumerate(test_questions, 1):
            print(f"{i}. Q: {question}")
            print(f"   Expected: {expected}")
            
            # Test with reasoning enabled
            response_with_reasoning = language_model.generate_response(
                question,
                context_size=5,
                min_activation=0.1,
                use_reasoning=True
            )
            print(f"   A (with reasoning): {response_with_reasoning}")
            
            # Test without reasoning for comparison
            response_without_reasoning = language_model.generate_response(
                question,
                context_size=5,
                min_activation=0.1,
                use_reasoning=False
            )
            print(f"   A (without reasoning): {response_without_reasoning}")
            
            # Check if responses are different
            if response_with_reasoning != response_without_reasoning:
                print(f"   ✓ Reasoning made a difference!")
            else:
                print(f"   - Same response with/without reasoning")
            
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
        
        # Count reasoning neurons
        reasoning_neurons = 0
        for neuron in container.graph.neurons.values():
            if hasattr(neuron, 'semantic_tags'):
                tags = neuron.semantic_tags
                if any(tag in ['reasoning', 'logic', 'analysis'] for tag in tags):
                    reasoning_neurons += 1
        
        print(f"Reasoning neurons: {reasoning_neurons}")
        print()
        
    finally:
        container.shutdown()


if __name__ == "__main__":
    test_reasoning()
