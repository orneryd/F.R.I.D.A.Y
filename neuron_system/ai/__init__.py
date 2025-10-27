"""
AI module for the 3D Synaptic Neuron System.

Provides language models, neural inference, training, and pre-training capabilities.
"""

# Core Models
from neuron_system.ai.language_model import LanguageModel
from neuron_system.ai.pretraining import PreTrainingLoader, EnglishKnowledgeBase
from neuron_system.ai.dataset_loader import DatasetLoader

# Training (consolidated)
from neuron_system.ai.training import (
    SmartTrainer,
    IncrementalTrainer,
    SelfTraining,
)

# Neural Inference (optional - requires transformers)
try:
    from neuron_system.ai.smart_language_model import SmartLanguageModel
    from neuron_system.ai.neural_inference import NeuralInferenceEngine
    from neuron_system.ai.model_loader import ModelLoader
    
    __all__ = [
        # Models
        "LanguageModel",
        "SmartLanguageModel",
        # Training
        "SmartTrainer",
        "IncrementalTrainer",
        "SelfTraining",
        # Neural Inference
        "NeuralInferenceEngine",
        "ModelLoader",
        # Data
        "PreTrainingLoader",
        "EnglishKnowledgeBase",
        "DatasetLoader",
    ]
except ImportError:
    # transformers not installed - neural inference not available
    __all__ = [
        # Models
        "LanguageModel",
        # Training
        "SmartTrainer",
        "IncrementalTrainer",
        "SelfTraining",
        # Data
        "PreTrainingLoader",
        "EnglishKnowledgeBase",
        "DatasetLoader",
    ]
