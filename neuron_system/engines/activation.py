"""
Activation propagation logic for the neuron network.

This module provides functions for propagating activation through synapses,
implementing the core mechanism of how information flows through the network.
"""

import logging
from typing import List, Dict, Set
from uuid import UUID
from collections import defaultdict

from neuron_system.core.graph import NeuronGraph
from neuron_system.core.neuron import Neuron

# Configure logger
logger = logging.getLogger(__name__)


def propagate_activation(
    graph: NeuronGraph,
    initial_neurons: List['ActivatedNeuron'],
    depth: int = 3,
    threshold: float = 0.1,
    decay_factor: float = 0.9
) -> List['ActivatedNeuron']:
    """
    Propagate activation through the neuron network via synapses.
    
    Implements iterative activation propagation where activation flows
    from source neurons to connected neurons through weighted synapses.
    
    Process:
    1. Start with initially activated neurons
    2. For each propagation depth:
        - For each activated neuron above threshold
        - Traverse outgoing synapses
        - Apply synapse weight to calculate target activation
        - Accumulate activation at target neurons
    3. Filter by threshold and return all activated neurons
    
    Args:
        graph: The neuron graph containing neurons and synapses
        initial_neurons: List of initially activated neurons
        depth: Number of propagation hops (default: 3)
        threshold: Minimum activation level to propagate (default: 0.1)
        decay_factor: Activation decay per hop (default: 0.9)
    
    Returns:
        List of all activated neurons after propagation
    
    Requirements: 5.2, 5.3, 15.1
    """
    # Import here to avoid circular dependency
    from neuron_system.engines.query import ActivatedNeuron
    
    # Track activation levels for all neurons
    activation_map: Dict[UUID, float] = {}
    neuron_map: Dict[UUID, Neuron] = {}
    metadata_map: Dict[UUID, Dict] = {}
    
    # Initialize with initial neurons
    for activated in initial_neurons:
        neuron_id = activated.neuron.id
        activation_map[neuron_id] = activated.activation
        neuron_map[neuron_id] = activated.neuron
        metadata_map[neuron_id] = {
            "initial_activation": activated.activation,
            "propagation_depth": 0,
            "sources": ["initial"]
        }
    
    logger.debug(
        f"Starting activation propagation with {len(initial_neurons)} "
        f"initial neurons, depth={depth}, threshold={threshold}"
    )
    
    # Propagate activation iteratively
    for current_depth in range(1, depth + 1):
        # Get neurons to propagate from (above threshold)
        source_neurons = [
            (neuron_id, activation)
            for neuron_id, activation in activation_map.items()
            if activation >= threshold
        ]
        
        if not source_neurons:
            logger.debug(f"No neurons above threshold at depth {current_depth}")
            break
        
        logger.debug(
            f"Depth {current_depth}: Propagating from {len(source_neurons)} neurons"
        )
        
        # Track new activations for this depth
        new_activations: Dict[UUID, float] = defaultdict(float)
        new_metadata: Dict[UUID, List] = defaultdict(list)
        
        # Propagate from each source neuron
        for source_id, source_activation in source_neurons:
            # Get outgoing synapses
            synapses = graph.get_outgoing_synapses(source_id)
            
            for synapse in synapses:
                # Mark synapse as traversed
                synapse.traverse()
                
                # Calculate target activation
                # activation = source_activation * synapse_weight * decay_factor
                target_activation = (
                    source_activation * 
                    abs(synapse.weight) * 
                    (decay_factor ** current_depth)
                )
                
                # Skip if below threshold
                if target_activation < threshold:
                    continue
                
                target_id = synapse.target_neuron_id
                
                # Accumulate activation at target
                new_activations[target_id] += target_activation
                
                # Track metadata
                new_metadata[target_id].append({
                    "source_neuron_id": str(source_id),
                    "synapse_id": str(synapse.id),
                    "synapse_weight": synapse.weight,
                    "contribution": target_activation,
                    "depth": current_depth
                })
        
        # Update activation map with new activations
        for target_id, new_activation in new_activations.items():
            # Get existing activation (if any)
            existing_activation = activation_map.get(target_id, 0.0)
            
            # Combine activations (sum with saturation at 1.0)
            combined_activation = min(1.0, existing_activation + new_activation)
            
            activation_map[target_id] = combined_activation
            
            # Store neuron reference if not already stored
            if target_id not in neuron_map:
                target_neuron = graph.get_neuron(target_id)
                if target_neuron:
                    neuron_map[target_id] = target_neuron
            
            # Update metadata
            if target_id not in metadata_map:
                metadata_map[target_id] = {
                    "initial_activation": 0.0,
                    "propagation_depth": current_depth,
                    "sources": []
                }
            
            metadata_map[target_id]["propagation_depth"] = max(
                metadata_map[target_id].get("propagation_depth", 0),
                current_depth
            )
            metadata_map[target_id]["sources"].extend(new_metadata[target_id])
        
        logger.debug(
            f"Depth {current_depth}: Activated {len(new_activations)} new neurons"
        )
    
    # Build result list with all activated neurons above threshold
    result = []
    for neuron_id, activation in activation_map.items():
        if activation >= threshold:
            neuron = neuron_map.get(neuron_id)
            if neuron:
                # Import here to avoid circular dependency
                from neuron_system.engines.query import ActivatedNeuron
                
                activated = ActivatedNeuron(
                    neuron=neuron,
                    activation=activation,
                    metadata=metadata_map.get(neuron_id, {})
                )
                result.append(activated)
    
    logger.info(
        f"Activation propagation complete: "
        f"{len(initial_neurons)} initial -> {len(result)} total activated neurons"
    )
    
    return result


def propagate_with_threshold_filtering(
    graph: NeuronGraph,
    initial_neurons: List['ActivatedNeuron'],
    depth: int = 3,
    min_threshold: float = 0.1,
    max_threshold: float = 1.0
) -> List['ActivatedNeuron']:
    """
    Propagate activation with threshold filtering at each level.
    
    This variant filters neurons at each propagation level to only
    include those within a specific activation range.
    
    Args:
        graph: The neuron graph
        initial_neurons: Initially activated neurons
        depth: Propagation depth
        min_threshold: Minimum activation to include
        max_threshold: Maximum activation to include
    
    Returns:
        List of activated neurons within threshold range
    """
    # Use standard propagation
    all_activated = propagate_activation(
        graph,
        initial_neurons,
        depth=depth,
        threshold=min_threshold
    )
    
    # Filter by threshold range
    filtered = [
        activated for activated in all_activated
        if min_threshold <= activated.activation <= max_threshold
    ]
    
    logger.debug(
        f"Threshold filtering: {len(all_activated)} -> {len(filtered)} neurons "
        f"(range: {min_threshold} to {max_threshold})"
    )
    
    return filtered


def calculate_activation_statistics(
    activated_neurons: List['ActivatedNeuron']
) -> Dict[str, float]:
    """
    Calculate statistics about activation levels.
    
    Args:
        activated_neurons: List of activated neurons
    
    Returns:
        Dictionary with activation statistics
    """
    if not activated_neurons:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0
        }
    
    import numpy as np
    
    activations = [n.activation for n in activated_neurons]
    
    return {
        "count": len(activations),
        "mean": float(np.mean(activations)),
        "median": float(np.median(activations)),
        "min": float(np.min(activations)),
        "max": float(np.max(activations)),
        "std": float(np.std(activations))
    }


def get_activation_paths(
    activated_neuron: 'ActivatedNeuron',
    max_paths: int = 5
) -> List[List[Dict]]:
    """
    Extract activation paths from metadata.
    
    Shows how activation reached a particular neuron through
    the network.
    
    Args:
        activated_neuron: Neuron to trace paths for
        max_paths: Maximum number of paths to return
    
    Returns:
        List of paths, where each path is a list of step dictionaries
    """
    sources = activated_neuron.metadata.get("sources", [])
    
    if not sources:
        return [[{
            "neuron_id": str(activated_neuron.neuron.id),
            "activation": activated_neuron.activation,
            "depth": 0,
            "source": "initial"
        }]]
    
    # Build paths from sources
    paths = []
    for source in sources[:max_paths]:
        path = [{
            "neuron_id": str(activated_neuron.neuron.id),
            "activation": activated_neuron.activation,
            "depth": source.get("depth", 0),
            "source_neuron_id": source.get("source_neuron_id"),
            "synapse_id": source.get("synapse_id"),
            "synapse_weight": source.get("synapse_weight"),
            "contribution": source.get("contribution")
        }]
        paths.append(path)
    
    return paths


def apply_activation_decay(
    activated_neurons: List['ActivatedNeuron'],
    decay_rate: float = 0.01
) -> List['ActivatedNeuron']:
    """
    Apply time-based decay to activation levels.
    
    This can be used to simulate activation decay over time,
    useful for temporal dynamics.
    
    Args:
        activated_neurons: Neurons with current activation
        decay_rate: Rate of decay per time step (0.0 to 1.0)
    
    Returns:
        Neurons with decayed activation levels
    """
    for activated in activated_neurons:
        activated.activation = max(0.0, activated.activation * (1.0 - decay_rate))
    
    return activated_neurons


def boost_activation_by_type(
    activated_neurons: List['ActivatedNeuron'],
    neuron_type_boosts: Dict[str, float]
) -> List['ActivatedNeuron']:
    """
    Boost activation levels based on neuron type.
    
    Useful for prioritizing certain types of neurons in results.
    
    Args:
        activated_neurons: Neurons to boost
        neuron_type_boosts: Dict mapping neuron type to boost factor
                           (e.g., {"tool": 1.2, "knowledge": 1.0})
    
    Returns:
        Neurons with boosted activation levels
    """
    for activated in activated_neurons:
        neuron_type = activated.neuron.neuron_type.value
        boost = neuron_type_boosts.get(neuron_type, 1.0)
        activated.activation = min(1.0, activated.activation * boost)
    
    return activated_neurons
