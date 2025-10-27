"""
Core Tests - Neurons, Synapses, Graph

Konsolidierte Tests für Core-Funktionalität.
"""

import pytest
import numpy as np
from uuid import UUID

from neuron_system.core.neuron import Neuron, NeuronType
from neuron_system.core.synapse import Synapse
from neuron_system.core.graph import NeuronGraph
from neuron_system.core.vector3d import Vector3D
from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron


class TestNeuron:
    """Test Neuron functionality."""
    
    def test_neuron_creation(self):
        """Test basic neuron creation."""
        neuron = KnowledgeNeuron(source_data="test")
        assert neuron.id is not None
        assert isinstance(neuron.id, UUID)
    
    def test_neuron_vector(self):
        """Test neuron vector."""
        neuron = KnowledgeNeuron(source_data="test")
        # Use dynamic dimension (default from compression engine)
        dim = 384  # This will be auto-detected in real usage
        neuron.vector = np.random.rand(dim)
        assert neuron.vector is not None
        assert len(neuron.vector) == dim
    
    def test_neuron_position(self):
        """Test neuron position."""
        neuron = KnowledgeNeuron(source_data="test")
        neuron.position = Vector3D(1.0, 2.0, 3.0)
        assert neuron.position.x == 1.0
        assert neuron.position.y == 2.0
        assert neuron.position.z == 3.0


class TestSynapse:
    """Test Synapse functionality."""
    
    def test_synapse_creation(self):
        """Test synapse creation."""
        neuron1 = KnowledgeNeuron(source_data="test1")
        neuron2 = KnowledgeNeuron(source_data="test2")
        
        synapse = Synapse(
            source_neuron_id=neuron1.id,
            target_neuron_id=neuron2.id,
            weight=0.5
        )
        
        assert synapse.id is not None
        assert synapse.weight == 0.5
    
    def test_synapse_weight_bounds(self):
        """Test synapse weight bounds."""
        neuron1 = KnowledgeNeuron(source_data="test1")
        neuron2 = KnowledgeNeuron(source_data="test2")
        
        synapse = Synapse(
            source_neuron_id=neuron1.id,
            target_neuron_id=neuron2.id,
            weight=1.5  # Should be clamped
        )
        
        assert -1.0 <= synapse.weight <= 1.0


class TestGraph:
    """Test NeuronGraph functionality."""
    
    def test_graph_creation(self):
        """Test graph creation."""
        graph = NeuronGraph()
        assert graph is not None
        assert len(graph.neurons) == 0
    
    def test_add_neuron(self):
        """Test adding neuron to graph."""
        graph = NeuronGraph()
        neuron = KnowledgeNeuron(source_data="test")
        neuron.position = Vector3D(0, 0, 0)
        # Dynamic dimension
        dim = 384
        neuron.vector = np.random.rand(dim)
        
        graph.add_neuron(neuron)
        assert len(graph.neurons) == 1
        assert neuron.id in graph.neurons
    
    def test_add_synapse(self):
        """Test adding synapse to graph."""
        graph = NeuronGraph()
        
        # Dynamic dimension
        dim = 384
        
        neuron1 = KnowledgeNeuron(source_data="test1")
        neuron1.position = Vector3D(0, 0, 0)
        neuron1.vector = np.random.rand(dim)
        
        neuron2 = KnowledgeNeuron(source_data="test2")
        neuron2.position = Vector3D(1, 1, 1)
        neuron2.vector = np.random.rand(dim)
        
        graph.add_neuron(neuron1)
        graph.add_neuron(neuron2)
        
        synapse = Synapse(
            source_neuron_id=neuron1.id,
            target_neuron_id=neuron2.id,
            weight=0.5
        )
        
        graph.add_synapse(synapse)
        assert len(graph.synapses) == 1
    
    def test_get_neuron(self):
        """Test getting neuron from graph."""
        graph = NeuronGraph()
        neuron = KnowledgeNeuron(source_data="test")
        neuron.position = Vector3D(0, 0, 0)
        # Dynamic dimension
        dim = 384
        neuron.vector = np.random.rand(dim)
        
        graph.add_neuron(neuron)
        
        retrieved = graph.get_neuron(neuron.id)
        assert retrieved is not None
        assert retrieved.id == neuron.id
    
    def test_remove_neuron(self):
        """Test removing neuron from graph."""
        graph = NeuronGraph()
        neuron = KnowledgeNeuron(source_data="test")
        neuron.position = Vector3D(0, 0, 0)
        # Dynamic dimension
        dim = 384
        neuron.vector = np.random.rand(dim)
        
        graph.add_neuron(neuron)
        assert len(graph.neurons) == 1
        
        graph.remove_neuron(neuron.id)
        assert len(graph.neurons) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
