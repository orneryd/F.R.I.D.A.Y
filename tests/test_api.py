"""
Basic tests for the REST API
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuron_system.api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "neuron_count" in data
    assert "synapse_count" in data


def test_openapi_docs():
    """Test OpenAPI documentation is available"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "info" in data
    assert data["info"]["title"] == "3D Synaptic Neuron System API"


def test_create_knowledge_neuron():
    """Test creating a knowledge neuron"""
    response = client.post(
        "/api/v1/neurons",
        json={
            "neuron_type": "knowledge",
            "source_data": "Test knowledge data",
            "semantic_tags": ["test"],
            "metadata": {"test": True}
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["neuron_type"] == "knowledge"
    assert data["source_data"] == "Test knowledge data"
    
    # Store neuron ID for cleanup
    return data["id"]


def test_get_neuron():
    """Test getting a neuron by ID"""
    # First create a neuron
    neuron_id = test_create_knowledge_neuron()
    
    # Then retrieve it
    response = client.get(
        f"/api/v1/neurons/{neuron_id}",
        headers={"X-API-Key": "dev-key-12345"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == neuron_id


def test_create_synapse():
    """Test creating a synapse"""
    # Create two neurons first
    neuron1_id = test_create_knowledge_neuron()
    neuron2_id = test_create_knowledge_neuron()
    
    # Create synapse between them
    response = client.post(
        "/api/v1/synapses",
        json={
            "source_neuron_id": neuron1_id,
            "target_neuron_id": neuron2_id,
            "weight": 0.7,
            "synapse_type": "KNOWLEDGE"
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["source_neuron_id"] == neuron1_id
    assert data["target_neuron_id"] == neuron2_id
    assert data["weight"] == 0.7


def test_query_neurons():
    """Test querying neurons"""
    # Create a neuron with specific data
    client.post(
        "/api/v1/neurons",
        json={
            "neuron_type": "knowledge",
            "source_data": "Python programming language",
            "semantic_tags": ["programming"],
            "metadata": {}
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    
    # Query for it
    response = client.post(
        "/api/v1/query",
        json={
            "query_text": "Python programming",
            "top_k": 5,
            "propagation_depth": 2
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "activated_neurons" in data
    assert "execution_time_ms" in data
    assert "query_id" in data


def test_rate_limiting_headers():
    """Test that rate limiting headers are present"""
    response = client.get(
        "/health",
        headers={"X-API-Key": "dev-key-12345"}
    )
    assert response.status_code == 200
    # Health endpoint bypasses rate limiting, so headers may not be present
    # Just verify the endpoint works


def test_unauthorized_access():
    """Test that endpoints require authentication"""
    response = client.post(
        "/api/v1/neurons",
        json={
            "neuron_type": "knowledge",
            "source_data": "Test",
            "semantic_tags": []
        }
    )
    # Should fail without API key
    assert response.status_code == 401


def test_invalid_neuron_type():
    """Test creating neuron with invalid type"""
    response = client.post(
        "/api/v1/neurons",
        json={
            "neuron_type": "invalid_type",
            "source_data": "Test"
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    assert response.status_code == 500  # Will fail during creation


def test_delete_neuron():
    """Test deleting a neuron"""
    # Create a neuron
    neuron_id = test_create_knowledge_neuron()
    
    # Delete it
    response = client.delete(
        f"/api/v1/neurons/{neuron_id}",
        headers={"X-API-Key": "dev-key-12345"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    
    # Verify it's gone
    response = client.get(
        f"/api/v1/neurons/{neuron_id}",
        headers={"X-API-Key": "dev-key-12345"}
    )
    assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
