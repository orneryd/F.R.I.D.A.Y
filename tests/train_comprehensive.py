"""
Comprehensive training script for the AI.

This script trains the AI with a large amount of data from multiple sources.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel
from neuron_system.ai.dataset_loader import DatasetLoader
from neuron_system.ai.pretraining import PreTrainingLoader


def main():
    print("=" * 70)
    print("COMPREHENSIVE AI TRAINING")
    print("=" * 70)
    print()
    print("This will train the AI with substantial data.")
    print("Estimated time: 10-30 minutes depending on your connection")
    print()
    
    response = input("Continue? (yes/no): ")
    if response.lower() != "yes":
        print("Cancelled.")
        return
    
    print()
    print("Starting training...")
    print("=" * 70)
    print()
    
    # Initialize with larger bounds for more neurons
    settings = Settings(
        database_path="comprehensive_ai.db",
        spatial_bounds_min=(-500.0, -500.0, -500.0),
        spatial_bounds_max=(500.0, 500.0, 500.0),
        enable_auto_save=True,
        auto_save_interval_seconds=300  # Save every 5 minutes
    )
    
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        # Create language model
        print("1. Initializing language model...")
        language_model = LanguageModel(
            container.graph,
            container.compression_engine,
            container.query_engine,
            container.training_engine
        )
        print("   [OK] Language model ready")
        print()
        
        # Load built-in knowledge
        print("2. Loading built-in English knowledge...")
        pre_loader = PreTrainingLoader(language_model, spatial_distribution="clustered")
        built_in_count = pre_loader.load_english_knowledge(
            create_connections=False,
            batch_size=20
        )
        print(f"   [OK] Loaded {built_in_count} built-in items")
        
        # Load additional knowledge
        print("   Loading additional knowledge...")
        from neuron_system.ai.additional_knowledge import AdditionalKnowledge
        additional_items = AdditionalKnowledge.get_all_knowledge()
        for item in additional_items:
            language_model.learn(
                text=item['text'],
                tags=item['tags'],
                create_connections=False
            )
        print(f"   [OK] Loaded {len(additional_items)} additional items")
        
        # Load reasoning knowledge
        print("   Loading reasoning knowledge...")
        from neuron_system.ai.reasoning_knowledge import ReasoningKnowledge
        reasoning_items = ReasoningKnowledge.get_all_knowledge()
        for item in reasoning_items:
            language_model.learn(
                text=item['text'],
                tags=item['tags'],
                create_connections=False
            )
        print(f"   [OK] Loaded {len(reasoning_items)} reasoning items")
        
        # Load conversation knowledge
        print("   Loading conversation knowledge...")
        from neuron_system.ai.conversation_knowledge import ConversationKnowledge
        conversation_items = ConversationKnowledge.get_all_knowledge()
        for item in conversation_items:
            language_model.learn(
                text=item['text'],
                tags=item['tags'],
                create_connections=False
            )
        print(f"   [OK] Loaded {len(conversation_items)} conversation items")
        print()
        
        # Create dataset loader
        loader = DatasetLoader(language_model)
        
        # Load from HuggingFace datasets
        print("3. Loading from HuggingFace datasets...")
        print("   This will download data - please be patient")
        print()
        
        # Wikitext (good quality, curated text)
        print("   3.1. Loading Wikitext...")
        wikitext_count = loader.load_from_huggingface(
            dataset_name="wikitext",
            split="train",
            text_field="text",
            max_samples=5000,  # Increased to 5000 samples for comprehensive knowledge base
            batch_size=100
        )
        print(f"       [OK] Loaded {wikitext_count} items from Wikitext")
        print()
        
        # Skip BookCorpus and C4 (deprecated dataset scripts)
        print("   3.2. Skipping BookCorpus (deprecated dataset script)")
        print("   3.3. Skipping C4 (deprecated dataset script)")
        book_count = 0
        c4_count = 0
        print()
        
        # Show progress
        stats = loader.get_statistics()
        print(f"   Total loaded so far: {stats['total_loaded']} items")
        print(f"   Total neurons: {stats['neurons_in_graph']}")
        print()
        
        # Create connections (optimized for speed)
        print("4. Creating connections between neurons...")
        print("   This will take a few minutes...")
        total_neurons = stats['neurons_in_graph']
        print(f"   (Creating connections for {total_neurons} neurons)")
        
        # Only create connections for a subset to speed up training
        # We'll create connections for the most important neurons (built-in knowledge)
        if total_neurons > 2000:
            print("   [OPTIMIZATION] Creating connections for first 2000 neurons for better quality")
            loader.create_connections_batch(
                top_k=3,  # 3 connections per neuron for good connectivity
                save_interval=500,
                max_neurons=2000  # Increased to 2000 neurons for much better coverage
            )
        else:
            loader.create_connections_batch(
                top_k=3,
                save_interval=200
            )
        print("   [OK] Connections created")
        print()
        
        # Final save
        print("5. Saving to database...")
        loader.save_to_database()
        print("   [OK] All data saved")
        print()
        
        # Final statistics
        print("=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        final_stats = loader.get_statistics()
        print(f"Total items loaded: {final_stats['total_loaded']}")
        print(f"Total neurons: {final_stats['neurons_in_graph']}")
        print(f"Total synapses: {final_stats['synapses_in_graph']}")
        print(f"Average connectivity: {final_stats['synapses_in_graph'] / final_stats['neurons_in_graph']:.2f}")
        print()
        print(f"Database saved to: {settings.database_path}")
        print()
        
        # Test the AI
        print("=" * 70)
        print("TESTING THE TRAINED AI")
        print("=" * 70)
        print()
        
        test_questions = [
            "What is knowledge?",
            "Tell me about language",
            "What is learning?",
            "Explain communication",
            "What is information?",
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"{i}. Q: {question}")
            response = language_model.generate_response(
                question,
                context_size=3,
                min_activation=0.1
            )
            print(f"   A: {response}")
            print()
        
        print("=" * 70)
        print("You can now use this trained AI with:")
        print(f"  python examples/example_load_trained_ai.py")
        print("  (Update the database path to: {settings.database_path})")
        print("=" * 70)
        
    finally:
        print()
        print("Shutting down...")
        container.shutdown()
        print("Done!")


if __name__ == "__main__":
    main()
