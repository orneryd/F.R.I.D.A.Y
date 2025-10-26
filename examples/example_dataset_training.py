"""
Example: Training the AI with real datasets

This example shows how to load real-world datasets and train
the neuron system with them.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel
from neuron_system.ai.dataset_loader import DatasetLoader


def main():
    """Main example function."""
    print("=" * 70)
    print("Training AI with Real Datasets")
    print("=" * 70)
    print()
    
    # Initialize application container
    print("1. Initializing system...")
    settings = Settings(
        database_path="trained_ai.db",
        spatial_bounds_min=(-200.0, -200.0, -200.0),
        spatial_bounds_max=(200.0, 200.0, 200.0)
    )
    
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
        
        # Create dataset loader
        print("3. Creating dataset loader...")
        loader = DatasetLoader(language_model)
        print("   [OK] Dataset loader created")
        print()
        
        # Load datasets
        print("4. Loading datasets...")
        print("-" * 70)
        
        # Option 1: Load from built-in knowledge base first
        print("\nOption 1: Loading built-in English knowledge...")
        from neuron_system.ai.pretraining import PreTrainingLoader
        
        pre_loader = PreTrainingLoader(language_model, spatial_distribution="clustered")
        built_in_count = pre_loader.load_english_knowledge(
            create_connections=False,
            batch_size=20
        )
        print(f"   [OK] Loaded {built_in_count} built-in knowledge items")
        
        # Option 2: Try to load from HuggingFace dataset
        print("\nOption 2: Loading from HuggingFace dataset...")
        print("(This will download data - may take a moment)")
        
        # Use a simple, reliable dataset
        hf_count = loader.load_from_huggingface(
            dataset_name="wikitext",
            split="train",
            text_field="text",
            max_samples=50,  # Small sample
            batch_size=10
        )
        
        if hf_count == 0:
            print("   Note: HuggingFace dataset not loaded (this is OK for testing)")
        else:
            print(f"   [OK] Loaded {hf_count} items from HuggingFace")
        
        total_count = built_in_count + hf_count
        print(f"\n   Total: {total_count} knowledge items loaded")
        
        # Option 2: Load from text file (if you have one)
        # print("\nOption 2: Loading from text file...")
        # file_count = loader.load_from_text_file(
        #     file_path="path/to/your/textfile.txt",
        #     chunk_size=500,
        #     tags=["custom", "text"]
        # )
        # print(f"   [OK] Loaded {file_count} text chunks")
        
        # Option 3: Load from JSON file
        # print("\nOption 3: Loading from JSON...")
        # json_count = loader.load_from_json(
        #     file_path="path/to/your/data.json",
        #     text_field="text",
        #     tags_field="tags"
        # )
        # print(f"   [OK] Loaded {json_count} JSON items")
        
        print()
        
        # Create connections between neurons
        print("5. Creating connections between neurons...")
        loader.create_connections_batch(
            top_k=5,
            save_interval=500  # Save every 500 neurons
        )
        print("   [OK] Connections created")
        print()
        
        # Explicitly save everything to database
        print("5.1. Saving to database...")
        loader.save_to_database()
        print("   [OK] Data saved to database")
        print()
        
        # Show statistics
        print("6. System statistics:")
        print("-" * 70)
        stats = loader.get_statistics()
        print(f"   Total items loaded: {stats['total_loaded']}")
        print(f"   Neurons in graph: {stats['neurons_in_graph']}")
        print(f"   Synapses in graph: {stats['synapses_in_graph']}")
        print()
        
        # Test the trained AI
        print("7. Testing the trained AI:")
        print("-" * 70)
        
        test_questions = [
            "What is Wikipedia?",
            "Tell me about knowledge",
            "What is an encyclopedia?",
        ]
        
        for question in test_questions:
            print(f"\nQ: {question}")
            response = language_model.generate_response(
                question,
                context_size=3,
                min_activation=0.2
            )
            print(f"A: {response}")
        
        print()
        print("=" * 70)
        print("[OK] Training Complete!")
        print("=" * 70)
        print()
        print("The AI has been trained with real-world data.")
        print(f"Database saved to: {settings.database_path}")
        print()
        
    finally:
        # Cleanup
        print("Shutting down...")
        container.shutdown()
        print("Done!")


def load_more_data():
    """
    Example of loading more comprehensive datasets.
    
    WARNING: This will download and process large amounts of data!
    """
    print("=" * 70)
    print("Loading Comprehensive Datasets")
    print("=" * 70)
    print()
    print("WARNING: This will download large datasets!")
    print("Make sure you have:")
    print("  - Sufficient disk space (several GB)")
    print("  - Good internet connection")
    print("  - Time (this can take hours)")
    print()
    
    response = input("Continue? (yes/no): ")
    if response.lower() != "yes":
        print("Cancelled.")
        return
    
    # Initialize
    settings = Settings(database_path="large_trained_ai.db")
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        language_model = LanguageModel(
            container.graph,
            container.compression_engine,
            container.query_engine,
            container.training_engine
        )
        
        loader = DatasetLoader(language_model)
        
        # Load larger datasets
        print("\n1. Loading Wikipedia (10,000 articles)...")
        loader.load_wikipedia(max_articles=10000, batch_size=100)
        
        print("\n2. Loading BookCorpus sample...")
        loader.load_bookcorpus(max_samples=5000, batch_size=100)
        
        print("\n3. Loading Common Crawl sample...")
        loader.load_common_crawl_sample(max_samples=5000, batch_size=100)
        
        print("\n4. Creating connections...")
        loader.create_connections_batch(top_k=5)
        
        # Show final statistics
        stats = loader.get_statistics()
        print("\nFinal Statistics:")
        print(f"  Total items: {stats['total_loaded']}")
        print(f"  Neurons: {stats['neurons_in_graph']}")
        print(f"  Synapses: {stats['synapses_in_graph']}")
        
    finally:
        container.shutdown()


if __name__ == "__main__":
    # Run basic example
    main()
    
    # Uncomment to load more comprehensive data
    # load_more_data()
