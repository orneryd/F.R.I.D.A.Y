"""
3D Synaptic Neuron System

A novel knowledge representation system that models knowledge as a 3D network
of neurons and synapses with native tool integration.
"""

__version__ = "0.1.0"

# Core components
from neuron_system.core.vector3d import Vector3D
from neuron_system.core.neuron import Neuron, NeuronType, NeuronTypeRegistry
from neuron_system.core.synapse import Synapse, SynapseType
from neuron_system.core.graph import NeuronGraph

# Neuron types
from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
from neuron_system.neuron_types.tool_neuron import ToolNeuron
from neuron_system.neuron_types.memory_neuron import MemoryNeuron

# Engines
from neuron_system.engines.compression import CompressionEngine
from neuron_system.engines.query import QueryEngine
from neuron_system.engines.training import TrainingEngine

# Configuration
from neuron_system.config.settings import Settings, get_settings, update_settings

__all__ = [
    # Core
    "Vector3D",
    "Neuron",
    "NeuronType",
    "NeuronTypeRegistry",
    "Synapse",
    "SynapseType",
    "NeuronGraph",
    # Neuron types
    "KnowledgeNeuron",
    "ToolNeuron",
    "MemoryNeuron",
    # Engines
    "CompressionEngine",
    "QueryEngine",
    "TrainingEngine",
    # Configuration
    "Settings",
    "get_settings",
    "update_settings",
]
