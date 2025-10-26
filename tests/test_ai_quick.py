"""
Quick test of the AI language model with pre-trained English knowledge.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel
from neuron_system.ai.pretraining import PreTrainingLoader

def main():
    print("=" * 70)
    print("Quick AI Test")
    print("=" * 70)
    print()
    
    # Initialize
    print("1. Initializing system...")
    settings = Settings(database_path=":memory:")  # In-memory for quick test
    container = ApplicationContainer(settings)
    container.initialize()
    print("   [OK] System initialized")
    print()
    
    try:
        # Create language model
        print("2. Creating language model...")
        language_model = LanguageModel(
            graph=container.graph,
            compression_engine=container.compression_engine,
            query_engine=container.query_engine,
            training_engine=container.training_engine
        )
        print("   [OK] Language model created")
        print()
        
        # Load pre-trained knowledge
        print("3. Loading pre-trained English knowledge...")
        loader = PreTrainingLoader(language_model, spatial_distribution="clustered")
        neurons_created = loader.load_english_knowledge(create_connections=True, batch_size=20)
        print(f"   [OK] Loaded {neurons_created} knowledge neurons")
        print()
        
        # Show statistics
        print("4. System statistics:")
        stats = language_model.get_statistics()
        print(f"   - Total Neurons: {stats['total_neurons']}")
        print(f"   - Knowledge Neurons: {stats['knowledge_neurons']}")
        print(f"   - Total Synapses: {stats['total_synapses']}")
        print(f"   - Average Connectivity: {stats['average_connectivity']:.2f}")
        print()
        
        # Test questions
        print("5. Testing AI with questions:")
        print("-" * 70)
        
        test_questions = [
            "What is hello?",
            "Tell me about colors",
            "What does walk mean?",
            "Explain a sentence"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\nQuestion {i}: {question}")
            response = language_model.generate_response(question, context_size=3, min_activation=0.2)
            print(f"Answer: {response}")
            
            # Show activated neurons
            results = language_model.understand(question, top_k=3)
            if results:
                print(f"[Activated {len(results)} neurons: ", end="")
                print(", ".join([f"{r.activation:.2f}" for r in results[:3]]), end="")
                print("]")
        
        print()
        print("=" * 70)
        print("[OK] AI Test Complete!")
        print("=" * 70)
        
    finally:
        container.shutdown()

if __name__ == "__main__":
    main()
