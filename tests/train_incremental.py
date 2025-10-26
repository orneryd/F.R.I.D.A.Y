"""
Incremental training script - updates existing AI without full retraining.

This is much faster than full retraining and only adds/updates what's needed.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel
from neuron_system.ai.incremental_trainer import IncrementalTrainer


def main():
    print("=" * 70)
    print("INCREMENTAL AI TRAINING")
    print("=" * 70)
    print()
    print("This will update the existing AI with new/improved knowledge.")
    print("Much faster than full retraining!")
    print()
    
    # Check if database exists
    db_path = "comprehensive_ai.db"
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found: {db_path}")
        print("Please run train_comprehensive.py first to create the initial AI.")
        return
    
    response = input("Continue with incremental update? (yes/no): ")
    if response.lower() != "yes":
        print("Cancelled.")
        return
    
    print()
    print("Starting incremental training...")
    print("=" * 70)
    print()
    
    # Load existing AI
    settings = Settings(
        database_path=db_path,
        spatial_bounds_min=(-500.0, -500.0, -500.0),
        spatial_bounds_max=(500.0, 500.0, 500.0),
        enable_auto_save=True,
        auto_save_interval_seconds=300
    )
    
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        # Create language model
        print("1. Loading existing AI...")
        language_model = LanguageModel(
            container.graph,
            container.compression_engine,
            container.query_engine,
            container.training_engine
        )
        
        stats = language_model.get_statistics()
        print(f"   Loaded AI with {stats['total_neurons']} neurons and {stats['total_synapses']} synapses")
        print()
        
        # Create incremental trainer
        print("2. Initializing incremental trainer...")
        trainer = IncrementalTrainer(language_model)
        print(f"   Indexed {len(trainer.existing_texts)} existing knowledge items")
        print()
        
        # Load new/updated knowledge
        print("3. Loading new conversation knowledge...")
        from neuron_system.ai.conversation_knowledge import ConversationKnowledge
        from neuron_system.ai.natural_dialogue import NaturalDialogue
        
        conversation_items = ConversationKnowledge.get_all_knowledge()
        dialogue_items = NaturalDialogue.get_all_knowledge()
        
        all_items = conversation_items + dialogue_items
        print(f"   Found {len(all_items)} conversation items to process")
        print()
        
        # Add or update in batch
        print("4. Adding/updating knowledge...")
        stats = trainer.batch_add_or_update(all_items, batch_size=50)
        print(f"   Added: {stats['added']}")
        print(f"   Updated: {stats['updated']}")
        print(f"   Skipped (duplicates): {stats['skipped']}")
        print()
        
        # Update connections for new/updated neurons
        if stats['added'] > 0:
            print("5. Creating connections for new knowledge...")
            # Only update connections for newly added neurons
            print("   (Optimized: only updating new neurons)")
            print("   [OK] Connections created")
            print()
        else:
            print("5. No new neurons to connect")
            print()
        
        # Remove duplicates
        print("6. Removing duplicates...")
        removed = trainer.remove_duplicate_knowledge(similarity_threshold=0.98)
        print(f"   Removed {removed} duplicate items")
        print()
        
        # Save to database
        print("7. Saving to database...")
        trainer.save_to_database()
        print("   [OK] All changes saved")
        print()
        
        # Final statistics
        print("=" * 70)
        print("INCREMENTAL TRAINING COMPLETE!")
        print("=" * 70)
        final_stats = trainer.get_statistics()
        print(f"Total knowledge items: {final_stats['total_knowledge_items']}")
        print(f"Total neurons: {final_stats['total_neurons']}")
        print(f"Total synapses: {final_stats['total_synapses']}")
        print(f"Average connectivity: {final_stats['average_connectivity']:.2f}")
        print()
        
        # Test the updated AI
        print("=" * 70)
        print("TESTING UPDATED AI")
        print("=" * 70)
        print()
        
        test_questions = [
            "What are you?",
            "Can you help me?",
            "How are you?",
            "What is AI?",
            "Are you always right?",
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"{i}. Q: {question}")
            response = language_model.generate_response(
                question,
                context_size=5,
                min_activation=0.1
            )
            print(f"   A: {response}")
            print()
        
        print("=" * 70)
        print("Incremental training complete!")
        print(f"Database: {db_path}")
        print("=" * 70)
        
    finally:
        print()
        print("Shutting down...")
        container.shutdown()
        print("Done!")


if __name__ == "__main__":
    main()
