"""
Example: Pre-trained AI with English Language Knowledge

This example demonstrates how to create an AI with pre-trained
English language knowledge and interact with it.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel
from neuron_system.ai.pretraining import PreTrainingLoader


def main():
    """Main example function."""
    print("=" * 70)
    print("Pre-trained AI with English Language Knowledge")
    print("=" * 70)
    print()
    
    # Initialize application container
    print("Initializing system...")
    settings = Settings(
        database_path="pretrained_ai.db",
        spatial_bounds_min=(-100.0, -100.0, -100.0),
        spatial_bounds_max=(100.0, 100.0, 100.0)
    )
    
    container = ApplicationContainer(settings)
    container.initialize()
    print()
    
    try:
        # Create language model
        print("Creating language model...")
        language_model = LanguageModel(
            graph=container.graph,
            compression_engine=container.compression_engine,
            query_engine=container.query_engine,
            training_engine=container.training_engine
        )
        print()
        
        # Check if already pre-trained
        stats = language_model.get_statistics()
        if stats['knowledge_neurons'] == 0:
            print("Loading pre-trained English knowledge...")
            print("This may take a minute...")
            print()
            
            # Load pre-trained knowledge
            loader = PreTrainingLoader(
                language_model=language_model,
                spatial_distribution="clustered"
            )
            
            neurons_created = loader.load_english_knowledge(
                create_connections=True,
                batch_size=10
            )
            
            print(f"\n✓ Loaded {neurons_created} pre-trained neurons")
            print()
        else:
            print(f"✓ System already has {stats['knowledge_neurons']} knowledge neurons")
            print()
        
        # Display statistics
        stats = language_model.get_statistics()
        print("System Statistics:")
        print("-" * 70)
        print(f"  Total Neurons: {stats['total_neurons']}")
        print(f"  Knowledge Neurons: {stats['knowledge_neurons']}")
        print(f"  Total Synapses: {stats['total_synapses']}")
        print(f"  Average Connectivity: {stats['average_connectivity']:.2f}")
        print()
        
        # Interactive Q&A
        print("=" * 70)
        print("Interactive AI Chat")
        print("=" * 70)
        print("Ask questions in English. Type 'quit' to exit.")
        print()
        
        # Example questions
        example_questions = [
            "What is hello?",
            "Tell me about colors",
            "What does walk mean?",
            "Explain grammar",
            "What is a question?"
        ]
        
        print("Example questions you can ask:")
        for i, q in enumerate(example_questions, 1):
            print(f"  {i}. {q}")
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nGoodbye!")
                    break
                
                # Generate response
                print("\nAI: ", end="")
                response = language_model.generate_response(
                    query=user_input,
                    context_size=5,
                    min_activation=0.2
                )
                print(response)
                print()
                
                # Show understanding (activated neurons)
                results = language_model.understand(user_input, top_k=3)
                if results:
                    print(f"[Activated {len(results)} neurons with relevance: ", end="")
                    print(", ".join([f"{r.activation:.2f}" for r in results[:3]]))
                    print("]")
                print()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print()
        
    finally:
        # Cleanup
        print("\nShutting down...")
        container.shutdown()
        print("Done!")


def demo_mode():
    """Run in demo mode with pre-defined questions."""
    print("=" * 70)
    print("Pre-trained AI Demo Mode")
    print("=" * 70)
    print()
    
    # Initialize
    settings = Settings(database_path="pretrained_ai.db")
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        # Create language model
        language_model = LanguageModel(
            graph=container.graph,
            compression_engine=container.compression_engine,
            query_engine=container.query_engine,
            training_engine=container.training_engine
        )
        
        # Load knowledge if needed
        stats = language_model.get_statistics()
        if stats['knowledge_neurons'] == 0:
            print("Loading pre-trained knowledge...")
            loader = PreTrainingLoader(language_model, "clustered")
            loader.load_english_knowledge(create_connections=True)
            print()
        
        # Demo questions
        questions = [
            "What is hello?",
            "Tell me about colors",
            "What does walk mean?",
            "Explain a sentence",
            "What is thank you?",
            "Tell me about numbers",
            "What is morning?",
            "Explain verbs"
        ]
        
        print("Demo: Asking pre-defined questions")
        print("-" * 70)
        print()
        
        for i, question in enumerate(questions, 1):
            print(f"Question {i}: {question}")
            response = language_model.generate_response(question, context_size=3)
            print(f"Answer: {response}")
            print()
        
    finally:
        container.shutdown()


if __name__ == "__main__":
    # Check for demo mode
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_mode()
    else:
        main()
