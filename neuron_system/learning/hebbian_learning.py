"""
Hebbian Learning: Dynamic synapse formation during usage
"Neurons that fire together, wire together"
"""
import logging
from typing import List, Set
from uuid import UUID, uuid4
from datetime import datetime

from neuron_system.storage.neuron_store import NeuronStore
from neuron_system.storage.synapse_store import SynapseStore
from neuron_system.core.synapse import Synapse

logger = logging.getLogger(__name__)


class HebbianLearner:
    """
    Creates and strengthens synapses based on co-activation
    
    Instead of pre-connecting all neurons, connections form organically
    when neurons are used together in responses.
    """
    
    def __init__(self, neuron_store: NeuronStore, synapse_store: SynapseStore):
        self.neuron_store = neuron_store
        self.synapse_store = synapse_store
        
        # Thresholds
        self.min_activation_threshold = 0.3  # Minimum activation to form connections
        self.initial_synapse_weight = 0.1    # New synapses start weak
        self.max_synapse_weight = 1.0        # Maximum strength
        self.weight_increment = 0.05         # How much to strengthen per co-activation
    
    def process_activation(self, activated_neuron_ids: List[UUID]):
        """
        Process a set of co-activated neurons and form/strengthen connections
        
        Args:
            activated_neuron_ids: List of neuron IDs that fired together
        """
        if len(activated_neuron_ids) < 2:
            return  # Need at least 2 neurons to form connections
        
        logger.info(f"Processing Hebbian learning for {len(activated_neuron_ids)} co-activated neurons")
        
        # Get all neurons
        neurons = {str(nid): self.neuron_store.get(nid) for nid in activated_neuron_ids}
        
        # Filter by activation level
        active_neurons = {
            nid: n for nid, n in neurons.items() 
            if n and n.activation_level >= self.min_activation_threshold
        }
        
        if len(active_neurons) < 2:
            logger.debug("Not enough highly activated neurons for connection")
            return
        
        active_ids = list(active_neurons.keys())
        connections_formed = 0
        connections_strengthened = 0
        
        # Form connections between all co-activated neurons
        for i, source_id in enumerate(active_ids):
            for target_id in active_ids[i+1:]:
                # Check if connection already exists
                existing = self.synapse_store.list_by_neurons(
                    UUID(source_id), 
                    UUID(target_id)
                )
                
                if existing:
                    # Strengthen existing connection
                    synapse = existing[0]
                    old_weight = synapse.weight
                    synapse.weight = min(
                        self.max_synapse_weight,
                        synapse.weight + self.weight_increment
                    )
                    synapse.usage_count += 1
                    synapse.last_traversed = datetime.now()
                    synapse.modified_at = datetime.now()
                    
                    self.synapse_store.update(synapse)
                    connections_strengthened += 1
                    
                    logger.debug(f"Strengthened synapse {synapse.id}: {old_weight:.3f} -> {synapse.weight:.3f}")
                else:
                    # Create new connection
                    from neuron_system.core.synapse import SynapseType
                    synapse = Synapse(
                        id=uuid4(),
                        source_neuron_id=UUID(source_id),
                        target_neuron_id=UUID(target_id),
                        weight=self.initial_synapse_weight,
                        synapse_type=SynapseType.KNOWLEDGE,
                        metadata={'learning_method': 'hebbian', 'co-activation': True}
                    )
                    
                    self.synapse_store.create(synapse)
                    connections_formed += 1
                    
                    logger.debug(f"Formed new synapse between {source_id[:8]} -> {target_id[:8]}")
        
        logger.info(f"Hebbian learning: {connections_formed} new, {connections_strengthened} strengthened")
    
    def prune_weak_connections(self, threshold: float = 0.05):
        """
        Remove synapses that haven't been strengthened (unused connections)
        
        Args:
            threshold: Weight below which synapses are pruned
        """
        deleted = self.synapse_store.delete_weak_synapses(threshold)
        logger.info(f"Pruned {deleted} weak synapses (weight < {threshold})")
        return deleted
    
    def get_connection_stats(self):
        """Get statistics about the connection network"""
        with self.synapse_store.db.get_connection() as conn:
            # Total synapses
            cursor = conn.execute("SELECT COUNT(*) as count FROM synapses")
            total_synapses = cursor.fetchone()['count']
            
            # Average weight
            cursor = conn.execute("SELECT AVG(weight) as avg FROM synapses")
            avg_weight = cursor.fetchone()['avg'] or 0
            
            # Strong connections (weight > 0.5)
            cursor = conn.execute("SELECT COUNT(*) as count FROM synapses WHERE weight > 0.5")
            strong_connections = cursor.fetchone()['count']
            
            # Weak connections (weight < 0.1)
            cursor = conn.execute("SELECT COUNT(*) as count FROM synapses WHERE weight < 0.1")
            weak_connections = cursor.fetchone()['count']
            
            return {
                'total_synapses': total_synapses,
                'average_weight': avg_weight,
                'strong_connections': strong_connections,
                'weak_connections': weak_connections,
                'connection_strength_ratio': strong_connections / total_synapses if total_synapses > 0 else 0
            }
