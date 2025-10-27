"""
DEPRECATED: Use neuron_system.ai.training.SmartTrainer instead.

This file is kept for backward compatibility and will be removed in version 3.0.
"""

import warnings
from neuron_system.ai.training import SmartTrainer as _SmartTrainer

warnings.warn(
    "smart_trainer.py is deprecated. Use neuron_system.ai.training.SmartTrainer instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility
SmartTrainer = _SmartTrainer

__all__ = ['SmartTrainer']
