"""
Example: Loading a pre-trained AI from database

This example shows how to load an AI that was previously trained
with datasets and saved to a database.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel


def main():
    """Main example function."""
    print("=" * 70)
    print("Loading Pre-trained AI from Database")
    print("=" * 70)
    print()
    
    # Check if database exists
    db_path = "trained_ai.db"
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found: {db_path}")
        print()
        print("Please train the AI first by running:")
        print("  python examples/example_dataset_training.py")
        print()
        return
    
    # Initialize application container with existing database
    print("1. Loading from database...")
    settings = Settings(database_path=db_path)
    container = ApplicationContainer(settings)
    container.initialize()
    print("   [OK] System initialized")
    print()
    
    try:
        # Create language model (it will use the loaded neurons)
        print("2. Creating language model...")
        language_model = LanguageModel(
            graph=container.graph,
            compression_engine=container.compression_engine,
            query_engine=container.query_engine,
            training_engine=container.training_engine
        )
        print("   [OK] Language model created")
        print()
        
        # Show statistics
        print("3. Loaded AI statistics:")
        print("-" * 70)
        stats = language_model.get_statistics()
        print(f"   Total Neurons: {stats['total_neurons']}")
        print(f"   Knowledge Neurons: {stats['knowledge_neurons']}")
        print(f"   Total Synapses: {stats['synapses_in_graph']}")
        print(f"   Average Connectivity: {stats['average_connectivity']:.2f}")
        print()
        
        # Interactive Q&A
        print("4. Interactive Q&A")
        print("=" * 70)
        print("Ask questions to the trained AI. Type 'quit' to exit.")
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
        # Cleanup (will save any changes)
        print("\nShutting down...")
        container.shutdown()
        print("Done!")


def test_trained_ai():
    """
    Test the trained AI with predefined questions.
    """
    print("=" * 70)
    print("Testing Pre-trained AI")
    print("=" * 70)
    print()
    
    db_path = "trained_ai.db"
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found: {db_path}")
        return
    
    # Initialize
    settings = Settings(database_path=db_path)
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        language_model = LanguageModel(
            container.graph,
            container.compression_engine,
            container.query_engine,
            container.training_engine
        )
        
        # Show stats
        stats = language_model.get_statistics()
        print(f"Loaded AI with {stats['total_neurons']} neurons")
        print()
        
        # Test questions
        test_questions = [
            "What is Wikipedia?",
            "Tell me about knowledge",
            "What is an encyclopedia?",
            "Explain information",
            "What is learning?",
        ]
        
        print("Testing with predefined questions:")
        print("-" * 70)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Q: {question}")
            response = language_model.generate_response(
                question,
                context_size=3,
                min_activation=0.2
            )
            print(f"   A: {response}")
        
        print()
        print("=" * 70)
        print("Test Complete!")
        print("=" * 70)
        
    finally:
        container.shutdown()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_trained_ai()
    else:
        main()
