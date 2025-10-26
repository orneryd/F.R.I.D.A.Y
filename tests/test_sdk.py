"""
Tests for the Neuron System Python SDK

These tests verify that the SDK client works correctly with the API.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuron_system.api.app import app
from neuron_system.sdk import (
    NeuronSystemClient,
    NotFoundError,
    ValidationError,
    ConnectionError as SDKConnectionError
)

# Create a test client for the API
test_client = TestClient(app)


class MockSession:
    """Mock session that uses FastAPI TestClient"""
    def __init__(self, test_client):
        self.test_client = test_client
        self.headers = {}
    
    def request(self, method, url, json=None, params=None, timeout=None):
        """Mock request method"""
        # Extract path from full URL
        path = url.split("localhost:8000")[-1] if "localhost:8000" in url else url
        
        # Make request using test client
        if method == "GET":
            return self.test_client.get(path, params=params)
        elif method == "POST":
            return self.test_client.post(path, json=json)
        elif method == "DELETE":
            return self.test_client.delete(path)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def close(self):
        """Mock close method"""
        pass


@pytest.fixture
def sdk_client():
    """Create SDK client with mocked session for testing"""
    client = NeuronSystemClient(base_url="http://localhost:8000")
    # Replace session with mock that uses TestClient
    client.session = MockSession(test_client)
    return client


def test_health_check(sdk_client):
    """Test health check through SDK"""
    health = sdk_client.health_check()
    assert health["status"] == "healthy"
    assert "neuron_count" in health
    assert "synapse_count" in health


def test_add_knowledge(sdk_client):
    """Test adding knowledge through SDK"""
    neuron = sdk_client.add_knowledge(
        text="Python is a programming language",
        tags=["programming", "python"]
    )
    
    assert "id" in neuron
    assert neuron["neuron_type"] == "knowledge"
    assert neuron["source_data"] == "Python is a programming language"
    assert "python" in neuron["semantic_tags"]


def test_create_and_get_neuron(sdk_client):
    """Test creating and retrieving a neuron"""
    # Create neuron
    created = sdk_client.create_neuron(
        neuron_type="knowledge",
        source_data="Test knowledge"
    )
    neuron_id = created["id"]
    
    # Get neuron
    retrieved = sdk_client.get_neuron(neuron_id)
    assert retrieved["id"] == neuron_id
    assert retrieved["source_data"] == "Test knowledge"


def test_create_synapse(sdk_client):
    """Test creating a synapse through SDK"""
    # Create two neurons
    neuron1 = sdk_client.add_knowledge("Knowledge 1")
    neuron2 = sdk_client.add_knowledge("Knowledge 2")
    
    # Create synapse
    synapse = sdk_client.create_synapse(
        source_neuron_id=neuron1["id"],
        target_neuron_id=neuron2["id"],
        weight=0.8
    )
    
    assert "id" in synapse
    assert synapse["source_neuron_id"] == neuron1["id"]
    assert synapse["target_neuron_id"] == neuron2["id"]
    assert synapse["weight"] == 0.8


def test_connect_convenience_method(sdk_client):
    """Test the connect convenience method"""
    # Create two neurons
    neuron1 = sdk_client.add_knowledge("Knowledge A")
    neuron2 = sdk_client.add_knowledge("Knowledge B")
    
    # Connect them
    synapse = sdk_client.connect(
        from_neuron_id=neuron1["id"],
        to_neuron_id=neuron2["id"],
        strength=0.7
    )
    
    assert synapse["weight"] == 0.7


def test_search(sdk_client):
    """Test search convenience method"""
    # Add some knowledge
    sdk_client.add_knowledge("Machine learning is a subset of AI", tags=["AI", "ML"])
    sdk_client.add_knowledge("Deep learning uses neural networks", tags=["AI", "DL"])
    
    # Search
    results = sdk_client.search("machine learning", limit=5)
    
    assert isinstance(results, list)
    assert len(results) > 0
    assert "score" in results[0]
    assert "content" in results[0]


def test_query(sdk_client):
    """Test query method"""
    # Add knowledge
    sdk_client.add_knowledge("Natural language processing")
    
    # Query
    response = sdk_client.query(
        query_text="language processing",
        top_k=5,
        propagation_depth=2
    )
    
    assert "activated_neurons" in response
    assert "execution_time_ms" in response
    assert isinstance(response["activated_neurons"], list)


def test_get_neighbors(sdk_client):
    """Test getting neuron neighbors"""
    # Create neurons and connect them
    neuron1 = sdk_client.add_knowledge("Central concept")
    neuron2 = sdk_client.add_knowledge("Related concept")
    
    sdk_client.connect(neuron1["id"], neuron2["id"], strength=0.8)
    
    # Get neighbors
    neighbors = sdk_client.get_neighbors(neuron1["id"])
    
    assert "neighbors" in neighbors
    assert neighbors["count"] >= 1


def test_train(sdk_client):
    """Test training convenience method"""
    # Create neuron
    neuron = sdk_client.add_knowledge("Initial knowledge")
    
    # Train it
    result = sdk_client.train(
        neuron_id=neuron["id"],
        new_knowledge="Updated knowledge",
        learning_rate=0.2
    )
    
    assert result["success"] is True
    assert "operation_id" in result


def test_adjust_neuron(sdk_client):
    """Test adjusting neuron vector"""
    # Create neuron
    neuron = sdk_client.add_knowledge("Some knowledge")
    
    # Adjust it
    result = sdk_client.adjust_neuron(
        neuron_id=neuron["id"],
        target_text="New knowledge",
        learning_rate=0.1
    )
    
    assert result["success"] is True


def test_strengthen_weaken_connection(sdk_client):
    """Test strengthening and weakening connections"""
    # Create neurons and synapse
    neuron1 = sdk_client.add_knowledge("Concept 1")
    neuron2 = sdk_client.add_knowledge("Concept 2")
    synapse = sdk_client.connect(neuron1["id"], neuron2["id"], strength=0.5)
    
    # Strengthen
    result = sdk_client.strengthen_connection(synapse["id"], amount=0.1)
    assert result["success"] is True
    assert result["details"]["final_weight"] > 0.5
    
    # Weaken
    result = sdk_client.weaken_connection(synapse["id"], amount=0.05)
    assert result["success"] is True


def test_get_network_stats(sdk_client):
    """Test getting network statistics"""
    stats = sdk_client.get_network_stats()
    
    assert "status" in stats
    assert "neuron_count" in stats
    assert "synapse_count" in stats
    assert stats["status"] == "healthy"


def test_delete_neuron(sdk_client):
    """Test deleting a neuron"""
    # Create neuron
    neuron = sdk_client.add_knowledge("Temporary knowledge")
    neuron_id = neuron["id"]
    
    # Delete it
    result = sdk_client.delete_neuron(neuron_id)
    assert result["success"] is True
    
    # Verify it's gone
    with pytest.raises(NotFoundError):
        sdk_client.get_neuron(neuron_id)


def test_not_found_error(sdk_client):
    """Test NotFoundError is raised correctly"""
    with pytest.raises(NotFoundError):
        sdk_client.get_neuron("00000000-0000-0000-0000-000000000000")


def test_batch_create_neurons(sdk_client):
    """Test batch neuron creation"""
    neurons_data = [
        {"neuron_type": "knowledge", "source_data": f"Knowledge {i}"}
        for i in range(5)
    ]
    
    result = sdk_client.create_neurons_batch(neurons_data)
    
    assert result["count"] == 5
    assert len(result["created_ids"]) == 5


def test_context_manager(sdk_client):
    """Test SDK client as context manager"""
    with NeuronSystemClient(base_url="http://localhost:8000") as client:
        # Replace session with mock
        client.session = MockSession(test_client)
        
        # Use the client
        health = client.health_check()
        assert health["status"] == "healthy"
    
    # Client should be closed after exiting context


def test_spatial_query(sdk_client):
    """Test spatial query"""
    # Add neuron at specific position
    neuron = sdk_client.create_neuron(
        neuron_type="knowledge",
        source_data="Positioned knowledge",
        position=(10.0, 20.0, 30.0)
    )
    
    # Query nearby
    results = sdk_client.spatial_query(
        center=(10.0, 20.0, 30.0),
        radius=50.0
    )
    
    assert "activated_neurons" in results
    assert len(results["activated_neurons"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
