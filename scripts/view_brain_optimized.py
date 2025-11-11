"""
Optimized Brain Viewer - Shows only strongest connections for performance.

Usage:
    python view_brain_optimized.py --max-synapses 500 --min-weight 0.5
"""

import argparse
from neuron_system.storage.database import DatabaseManager
from neuron_system.storage.neuron_store import NeuronStore
from neuron_system.storage.synapse_store import SynapseStore
from neuron_system.visualization.live_brain_viewer import LiveBrainViewer


def main():
    parser = argparse.ArgumentParser(description="View Friday's brain with performance optimizations")
    parser.add_argument(
        '--database',
        default='data/neuron_system.db',
        help='Database path'
    )
    parser.add_argument(
        '--max-synapses',
        type=int,
        default=1000,
        help='Maximum number of synapses to display (default: 1000)'
    )
    parser.add_argument(
        '--min-weight',
        type=float,
        default=0.3,
        help='Minimum synapse weight to display (default: 0.3)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port for web server (default: 5000)'
    )
    
    args = parser.parse_args()
    
    # Initialize
    db_manager = DatabaseManager(args.database)
    neuron_store = NeuronStore(db_manager)
    synapse_store = SynapseStore(db_manager)
    
    # Get stats
    neurons = neuron_store.list_all()
    all_synapses = synapse_store.list_all(limit=200000)
    
    print(f"""
================================================================
                OPTIMIZED BRAIN VIEWER
================================================================

Database: {args.database}

Current State:
   Neurons:        {len(neurons)}
   Total Synapses: {len(all_synapses)}

Display Settings:
   Max Synapses:   {args.max_synapses}
   Min Weight:     {args.min_weight}
   
This will show only the {args.max_synapses} strongest connections
with weight >= {args.min_weight} for better performance.

Starting server on http://localhost:{args.port}
Press Ctrl+C to stop
================================================================
""")
    
    # Start viewer with custom settings
    viewer = LiveBrainViewer(neuron_store, synapse_store, port=args.port)
    
    # Store settings in viewer instance
    viewer.default_max_synapses = args.max_synapses
    viewer.default_min_weight = args.min_weight
    
    viewer.run()


if __name__ == "__main__":
    main()
