"""
Test visualization endpoints
"""
import pytest
from fastapi.testclient import TestClient
from uuid import uuid4

from neuron_system.api.app import app, app_state
from neuron_system.core.graph import NeuronGraph
from neuron_system.core.vector3d import Vector3D
from neuron_system.core.synapse import Synapse
from neuron_system.engines.compression import CompressionEngine
from neuron_system.tools.tool_cluster import ToolCluster
import numpy as np


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def setup_graph():
    """Setup test graph with neurons and synapses"""
    # Initialize graph
    graph = NeuronGraph(bounds=(Vector3D(-100, -100, -100), Vector3D(100, 100, 100)))
    app_state.graph = graph
    
    # Initialize compression engine
    app_state.compression_engine = CompressionEngine()
    
    # Create test neurons
    neurons = []
    
    # Knowledge neurons
    for i in range(5):
        neuron = graph.create_neuron(
            neuron_type="knowledge",
            source_data=f"Test knowledge {i}",
            semantic_tags=["test", f"tag{i}"],
            positioning_strategy="random"
        )
        neuron.activation_level = 0.5 + (i * 0.1)
        neuron.vector = np.random.rand(384)
        neurons.append(neuron)
    
    # Tool neurons
    for i in range(3):
        neuron = graph.create_neuron(
            neuron_type="tool",
            function_signature=f"test_tool_{i}()",
            executable_code=f"return 'result_{i}'",
            input_schema={},
            output_schema={},
            positioning_strategy="random"
        )
        neuron.activation_level = 0.3 + (i * 0.1)
        neuron.vector = np.random.rand(384)
        neurons.append(neuron)
    
    # Create synapses
    synapses = []
    for i in range(len(neurons) - 1):
        synapse = Synapse()
        synapse.id = uuid4()
        synapse.source_neuron_id = neurons[i].id
        synapse.target_neuron_id = neurons[i + 1].id
        synapse.weight = 0.5 + (i * 0.1)
        synapse.synapse_type = "KNOWLEDGE"
        synapse.usage_count = i
        graph.add_synapse(synapse)
        synapses.append(synapse)
    
    # Create a tool cluster
    tool_neurons = [n for n in neurons if n.neuron_type.value == "tool"]
    if len(tool_neurons) >= 2:
        cluster = ToolCluster(
            name="test_cluster",
            tool_neurons=[tool_neurons[0].id, tool_neurons[1].id],
            execution_graph={
                tool_neurons[0].id: [tool_neurons[1].id],
                tool_neurons[1].id: []
            },
            input_interface={"input": "string"},
            output_interface={"output": "string"}
        )
        graph.add_cluster(cluster)
    
    yield graph
    
    # Cleanup
    graph.clear()


def test_get_visualization_neurons(client, setup_graph):
    """Test GET /visualization/neurons endpoint"""
    response = client.get("/api/v1/visualization/neurons")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "neurons" in data
    assert "count" in data
    assert "bounds" in data
    
    assert data["count"] > 0
    assert len(data["neurons"]) == data["count"]
    
    # Check neuron structure
    neuron = data["neurons"][0]
    assert "id" in neuron
    assert "neuron_type" in neuron
    assert "position" in neuron
    assert "activation_level" in neuron
    assert "color" in neuron
    assert "size" in neuron
    assert "label" in neuron
    
    # Check position structure
    assert "x" in neuron["position"]
    assert "y" in neuron["position"]
    assert "z" in neuron["position"]
    
    # Check bounds structure
    assert "min" in data["bounds"]
    assert "max" in data["bounds"]


def test_get_visualization_neurons_with_filters(client, setup_graph):
    """Test GET /visualization/neurons with filters"""
    # Filter by type
    response = client.get("/api/v1/visualization/neurons?neuron_type=knowledge")
    assert response.status_code == 200
    data = response.json()
    
    for neuron in data["neurons"]:
        assert neuron["neuron_type"] == "knowledge"
    
    # Filter by activation
    response = client.get("/api/v1/visualization/neurons?min_activation=0.6")
    assert response.status_code == 200
    data = response.json()
    
    for neuron in data["neurons"]:
        assert neuron["activation_level"] >= 0.6
    
    # Filter with limit
    response = client.get("/api/v1/visualization/neurons?limit=3")
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["neurons"]) <= 3


def test_get_visualization_synapses(client, setup_graph):
    """Test GET /visualization/synapses endpoint"""
    response = client.get("/api/v1/visualization/synapses")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "synapses" in data
    assert "count" in data
    
    assert data["count"] > 0
    assert len(data["synapses"]) == data["count"]
    
    # Check synapse structure
    synapse = data["synapses"][0]
    assert "id" in synapse
    assert "source_neuron_id" in synapse
    assert "target_neuron_id" in synapse
    assert "weight" in synapse
    assert "synapse_type" in synapse
    assert "color" in synapse
    assert "thickness" in synapse


def test_get_visualization_synapses_with_filters(client, setup_graph):
    """Test GET /visualization/synapses with filters"""
    # Filter by weight
    response = client.get("/api/v1/visualization/synapses?min_weight=0.6")
    assert response.status_code == 200
    data = response.json()
    
    for synapse in data["synapses"]:
        assert synapse["weight"] >= 0.6
    
    # Filter with limit
    response = client.get("/api/v1/visualization/synapses?limit=2")
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["synapses"]) <= 2


def test_get_visualization_clusters(client, setup_graph):
    """Test GET /visualization/clusters endpoint"""
    response = client.get("/api/v1/visualization/clusters")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "clusters" in data
    assert "count" in data
    
    if data["count"] > 0:
        # Check cluster structure
        cluster = data["clusters"][0]
        assert "cluster_id" in cluster
        assert "name" in cluster
        assert "neurons" in cluster
        assert "boundary" in cluster
        assert "metadata" in cluster
        
        # Check boundary structure
        boundary = cluster["boundary"]
        assert "center" in boundary
        assert "radius" in boundary
        assert "neuron_count" in boundary


def test_get_threejs_scene(client, setup_graph):
    """Test GET /visualization/threejs endpoint"""
    response = client.get("/api/v1/visualization/threejs")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "neurons" in data
    assert "synapses" in data
    assert "clusters" in data
    assert "bounds" in data
    assert "metadata" in data
    
    # Check metadata
    assert data["metadata"]["format"] == "threejs"
    assert data["metadata"]["neuronCount"] > 0
    
    # Check neuron structure (Three.js format)
    if data["neurons"]:
        neuron = data["neurons"][0]
        assert "id" in neuron
        assert "position" in neuron
        assert "color" in neuron
        assert "size" in neuron
        assert "type" in neuron
        assert "activation" in neuron
        
        # Check color is normalized (0.0 to 1.0)
        color = neuron["color"]
        assert 0.0 <= color["r"] <= 1.0
        assert 0.0 <= color["g"] <= 1.0
        assert 0.0 <= color["b"] <= 1.0
    
    # Check synapse structure (Three.js format)
    if data["synapses"]:
        synapse = data["synapses"][0]
        assert "id" in synapse
        assert "start" in synapse
        assert "end" in synapse
        assert "color" in synapse
        assert "thickness" in synapse
        assert "weight" in synapse
        
        # Check positions
        assert "x" in synapse["start"]
        assert "y" in synapse["start"]
        assert "z" in synapse["start"]
    
    # Check bounds structure
    bounds = data["bounds"]
    assert "min" in bounds
    assert "max" in bounds
    assert "center" in bounds
    assert "size" in bounds


def test_get_threejs_scene_with_filters(client, setup_graph):
    """Test GET /visualization/threejs with filters"""
    # Without synapses
    response = client.get("/api/v1/visualization/threejs?include_synapses=false")
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["synapses"]) == 0
    
    # Without clusters
    response = client.get("/api/v1/visualization/threejs?include_clusters=false")
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["clusters"]) == 0
    
    # With neuron type filter
    response = client.get("/api/v1/visualization/threejs?neuron_type=tool")
    assert response.status_code == 200
    data = response.json()
    
    for neuron in data["neurons"]:
        assert neuron["type"] == "tool"


def test_visualization_color_calculations():
    """Test color calculation functions"""
    from neuron_system.api.routes.visualization import (
        _get_neuron_color,
        _get_synapse_color,
        _hex_to_threejs_color
    )
    
    # Test neuron color
    color = _get_neuron_color("knowledge", 0.5)
    assert color.startswith("#")
    assert len(color) == 7
    
    # Test synapse color (positive weight)
    color = _get_synapse_color(0.8, "KNOWLEDGE")
    assert color.startswith("#")
    
    # Test synapse color (negative weight)
    color = _get_synapse_color(-0.5, "KNOWLEDGE")
    assert color.startswith("#")
    
    # Test hex to Three.js conversion
    threejs_color = _hex_to_threejs_color("#ff8800")
    assert 0.0 <= threejs_color.r <= 1.0
    assert 0.0 <= threejs_color.g <= 1.0
    assert 0.0 <= threejs_color.b <= 1.0


def test_visualization_size_calculations():
    """Test size calculation functions"""
    from neuron_system.api.routes.visualization import _get_neuron_size, _get_synapse_thickness
    from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
    
    # Create test neuron
    neuron = KnowledgeNeuron(source_data="test", semantic_tags=[])
    neuron.activation_level = 0.5
    
    # Test neuron size
    size = _get_neuron_size(neuron, connection_count=10)
    assert size > 0
    
    # Test synapse thickness
    thickness = _get_synapse_thickness(0.8)
    assert 0.1 <= thickness <= 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
