"""
Migration script to add importance attribute to existing neurons.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings


def migrate_add_importance():
    print("=" * 70)
    print("MIGRATING: Adding importance attribute to all neurons")
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
        total_neurons = len(container.graph.neurons)
        print(f"Found {total_neurons} neurons to migrate")
        print()
        
        migrated = 0
        for neuron_id, neuron in container.graph.neurons.items():
            # Add importance if it doesn't exist
            if not hasattr(neuron, 'importance'):
                neuron.importance = 0.5  # Default middle value
                migrated += 1
                
                if migrated % 1000 == 0:
                    print(f"Migrated {migrated}/{total_neurons} neurons...")
        
        print()
        print(f"✓ Migrated {migrated} neurons")
        print(f"✓ {total_neurons - migrated} neurons already had importance")
        print()
        
        # Save to database
        print("Saving to database...")
        if hasattr(container.graph, 'neuron_store'):
            neurons_to_update = list(container.graph.neurons.values())
            container.graph.neuron_store.batch_update(neurons_to_update)
            print(f"✓ Saved {len(neurons_to_update)} neurons to database")
        
        print()
        print("=" * 70)
        print("MIGRATION COMPLETE")
        print("=" * 70)
        
    finally:
        container.shutdown()


if __name__ == "__main__":
    migrate_add_importance()
