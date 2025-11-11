"""
Start the live brain viewer
Run this while training or assimilating to see real-time updates
"""
from neuron_system.storage.database import DatabaseManager
from neuron_system.storage.neuron_store import NeuronStore
from neuron_system.storage.synapse_store import SynapseStore
from neuron_system.visualization.live_brain_viewer import LiveBrainViewer


def main():
    print("=" * 60)
    print("FRIDAY AI - LIVE BRAIN VIEWER")
    print("=" * 60)
    print()
    print("Starting live visualization server...")
    print("The brain will update in real-time as you:")
    print("  - Train new neurons")
    print("  - Assimilate knowledge")
    print("  - Generate responses")
    print()
    
    # Initialize database and stores with correct path
    db_path = "data/neuron_system.db"
    db_manager = DatabaseManager(db_path)
    neuron_store = NeuronStore(db_manager)
    synapse_store = SynapseStore(db_manager)
    
    # Create and start live viewer
    viewer = LiveBrainViewer(neuron_store, synapse_store, port=5000)
    
    print("=" * 60)
    print("🚀 Server starting on http://localhost:5000")
    print("=" * 60)
    print()
    print("Open your browser and navigate to:")
    print("  👉 http://localhost:5000")
    print()
    print("Then run training or assimilation in another terminal:")
    print("  python train.py")
    print("  python assimilate.py")
    print()
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        viewer.run(debug=False)
    except KeyboardInterrupt:
        print("\n\nShutting down live viewer...")
        viewer.stop_monitoring()
        print("Goodbye!")


if __name__ == "__main__":
    main()
