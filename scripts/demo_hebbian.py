"""
Demo: Hebbian Learning in Action
Watch connections form as neurons fire together
"""
import time
from uuid import UUID
from neuron_system.storage.database import DatabaseManager
from neuron_system.storage.neuron_store import NeuronStore
from neuron_system.storage.synapse_store import SynapseStore
from neuron_system.learning.hebbian_learning import HebbianLearner


def main():
    print("=" * 60)
    print("HEBBIAN LEARNING DEMO")
    print("=" * 60)
    print()
    print("Simulating neuron co-activation patterns...")
    print("Watch connections form and strengthen!")
    print()
    
    # Initialize
    db = DatabaseManager('data/neuron_system.db')
    neuron_store = NeuronStore(db)
    synapse_store = SynapseStore(db)
    learner = HebbianLearner(neuron_store, synapse_store)
    
    # Get some neurons
    neurons = neuron_store.list_all(limit=20)
    if len(neurons) < 5:
        print("Error: Need at least 5 neurons in database")
        print("Run training first: python train.py")
        return
    
    print(f"Using {len(neurons)} neurons for demo")
    print()
    
    # Initial stats
    stats = learner.get_connection_stats()
    print("Initial state:")
    print(f"  Total synapses: {stats['total_synapses']}")
    print(f"  Average weight: {stats['average_weight']:.3f}")
    print(f"  Strong connections: {stats['strong_connections']}")
    print()
    
    # Simulate 5 "thoughts" (co-activation patterns)
    thoughts = [
        # Thought 1: Neurons 0, 1, 2 fire together
        [neurons[0].id, neurons[1].id, neurons[2].id],
        
        # Thought 2: Neurons 1, 2, 3 fire together (overlaps with thought 1)
        [neurons[1].id, neurons[2].id, neurons[3].id],
        
        # Thought 3: Neurons 0, 1, 2 fire again (strengthens connections)
        [neurons[0].id, neurons[1].id, neurons[2].id],
        
        # Thought 4: New pattern with neurons 4, 5, 6
        [neurons[4].id, neurons[5].id, neurons[6].id],
        
        # Thought 5: Neurons 0, 1, 2 fire AGAIN (further strengthening)
        [neurons[0].id, neurons[1].id, neurons[2].id],
    ]
    
    # Set activation levels (simulate firing)
    for i, neuron in enumerate(neurons[:7]):
        neuron.activation_level = 0.8  # High activation
        neuron_store.update(neuron)
    
    # Process each thought
    for i, thought in enumerate(thoughts, 1):
        print(f"Thought {i}: {len(thought)} neurons co-activate")
        learner.process_activation(thought)
        
        # Show stats after each thought
        stats = learner.get_connection_stats()
        print(f"  → Synapses: {stats['total_synapses']}, Avg weight: {stats['average_weight']:.3f}")
        print()
        time.sleep(0.5)
    
    # Final stats
    print("=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    stats = learner.get_connection_stats()
    print(f"Total synapses: {stats['total_synapses']}")
    print(f"Average weight: {stats['average_weight']:.3f}")
    print(f"Strong connections (>0.5): {stats['strong_connections']}")
    print(f"Weak connections (<0.1): {stats['weak_connections']}")
    print()
    
    print("Notice how:")
    print("  • Neurons 0,1,2 formed STRONG connections (fired 3 times)")
    print("  • Neurons 1,2,3 formed MEDIUM connections (fired 1 time)")
    print("  • Neurons 4,5,6 formed WEAK connections (fired 1 time)")
    print()
    print("This is Hebbian learning: 'Neurons that fire together, wire together!'")
    print()


if __name__ == "__main__":
    main()
