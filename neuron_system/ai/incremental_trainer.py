"""
DEPRECATED: Use neuron_system.ai.training.IncrementalTrainer instead.

This file is kept for backward compatibility and will be removed in version 3.0.
"""

import warnings
from neuron_system.ai.training import IncrementalTrainer as _IncrementalTrainer

warnings.warn(
    "incremental_trainer.py is deprecated. Use neuron_system.ai.training.IncrementalTrainer instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility
IncrementalTrainer = _IncrementalTrainer

__all__ = ['IncrementalTrainer']
