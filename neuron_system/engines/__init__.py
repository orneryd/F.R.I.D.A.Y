"""
Processing engines for compression, query, and training.
"""

from neuron_system.engines.compression import CompressionEngine
from neuron_system.engines.query import QueryEngine, ActivatedNeuron
from neuron_system.engines.activation import (
    propagate_activation,
    propagate_with_threshold_filtering,
    calculate_activation_statistics,
    get_activation_paths,
    apply_activation_decay,
    boost_activation_by_type
)

__all__ = [
    'CompressionEngine',
    'QueryEngine',
    'ActivatedNeuron',
    'propagate_activation',
    'propagate_with_threshold_filtering',
    'calculate_activation_statistics',
    'get_activation_paths',
    'apply_activation_decay',
    'boost_activation_by_type'
]
