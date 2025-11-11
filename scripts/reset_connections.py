"""
Reset neural connections to sparse initialization
Keeps neurons but removes most synapses, allowing organic growth
"""
from neuron_system.storage.database import DatabaseManager
from neuron_system.storage.synapse_store import SynapseStore
from neuron_system.storage.neuron_store import NeuronStore


def main():
    print("=" * 60)
    print("RESET NEURAL CONNECTIONS")
    print("=" * 60)
    print()
    print("This will:")
    print("  ✓ Keep all neurons (knowledge preserved)")
    print("  ✗ Remove ALL synapses (connections reset)")
    print("  → Connections will form organically during usage")
    print()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--force':
        print("Force mode: Proceeding without confirmation")
    else:
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return
    
    print()
    print("Connecting to database...")
    db = DatabaseManager('data/neuron_system.db')
    neuron_store = NeuronStore(db)
    synapse_store = SynapseStore(db)
    
    # Get current stats
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) as count FROM neurons")
        neuron_count = cursor.fetchone()['count']
        
        cursor = conn.execute("SELECT COUNT(*) as count FROM synapses")
        synapse_count = cursor.fetchone()['count']
    
    print(f"Current state:")
    print(f"  Neurons: {neuron_count}")
    print(f"  Synapses: {synapse_count}")
    print()
    
    # Delete all synapses
    print("Deleting all synapses...")
    with db.transaction() as conn:
        cursor = conn.execute("DELETE FROM synapses")
        deleted = cursor.rowcount
    
    print(f"✓ Deleted {deleted} synapses")
    print()
    
    # Verify
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) as count FROM synapses")
        remaining = cursor.fetchone()['count']
    
    print("=" * 60)
    print("RESET COMPLETE")
    print("=" * 60)
    print(f"Neurons: {neuron_count} (preserved)")
    print(f"Synapses: {remaining} (reset)")
    print()
    print("Next steps:")
    print("  1. Start live viewer: python live_viewer.py")
    print("  2. Use Friday AI: python main.py")
    print("  3. Watch connections form in real-time!")
    print()


if __name__ == "__main__":
    main()
