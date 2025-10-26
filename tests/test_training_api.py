"""
Tests for the training API endpoints
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuron_system.api.app import app

client = TestClient(app)


def create_test_neuron():
    """Helper to create a test neuron"""
    response = client.post(
        "/api/v1/neurons",
        json={
            "neuron_type": "knowledge",
            "source_data": "Test knowledge for training",
            "semantic_tags": ["test", "training"],
            "metadata": {}
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    assert response.status_code == 201
    return response.json()["id"]


def create_test_synapse(source_id, target_id):
    """Helper to create a test synapse"""
    response = client.post(
        "/api/v1/synapses",
        json={
            "source_neuron_id": source_id,
            "target_neuron_id": target_id,
            "weight": 0.5,
            "synapse_type": "KNOWLEDGE"
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_adjust_neuron_with_text():
    """Test adjusting a neuron using target text"""
    # Create a neuron
    neuron_id = create_test_neuron()
    
    # Adjust it using text
    response = client.post(
        "/api/v1/training/adjust-neuron",
        json={
            "neuron_id": neuron_id,
            "target_text": "Updated knowledge about machine learning",
            "learning_rate": 0.2
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "operation_id" in data
    assert data["details"]["neuron_id"] == neuron_id
    assert data["details"]["learning_rate"] == 0.2
    assert data["details"]["used_text_compression"] is True
    print(f"✓ Adjusted neuron with text: {data['message']}")


def test_adjust_neuron_with_vector():
    """Test adjusting a neuron using target vector"""
    # Create a neuron
    neuron_id = create_test_neuron()
    
    # Create a target vector (384 dimensions)
    target_vector = [0.1] * 384
    
    # Adjust it using vector
    response = client.post(
        "/api/v1/training/adjust-neuron",
        json={
            "neuron_id": neuron_id,
            "target_vector": target_vector,
            "learning_rate": 0.15
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["details"]["used_text_compression"] is False
    print(f"✓ Adjusted neuron with vector: {data['message']}")


def test_adjust_neuron_invalid_vector():
    """Test adjusting neuron with invalid vector dimensions"""
    neuron_id = create_test_neuron()
    
    # Invalid vector (wrong dimensions)
    response = client.post(
        "/api/v1/training/adjust-neuron",
        json={
            "neuron_id": neuron_id,
            "target_vector": [0.1] * 100,  # Wrong size
            "learning_rate": 0.1
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    
    assert response.status_code == 422  # Validation error
    print("✓ Correctly rejected invalid vector dimensions")


def test_strengthen_synapse():
    """Test strengthening a synapse"""
    # Create two neurons and a synapse
    neuron1_id = create_test_neuron()
    neuron2_id = create_test_neuron()
    synapse_id = create_test_synapse(neuron1_id, neuron2_id)
    
    # Strengthen the synapse
    response = client.post(
        "/api/v1/training/adjust-synapse",
        json={
            "synapse_id": synapse_id,
            "operation": "strengthen",
            "delta": 0.1
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["details"]["operation"] == "strengthen"
    assert data["details"]["initial_weight"] == 0.5
    assert data["details"]["final_weight"] == 0.6
    assert data["details"]["deleted"] is False
    print(f"✓ Strengthened synapse: {data['message']}")


def test_weaken_synapse():
    """Test weakening a synapse"""
    # Create two neurons and a synapse
    neuron1_id = create_test_neuron()
    neuron2_id = create_test_neuron()
    synapse_id = create_test_synapse(neuron1_id, neuron2_id)
    
    # Weaken the synapse
    response = client.post(
        "/api/v1/training/adjust-synapse",
        json={
            "synapse_id": synapse_id,
            "operation": "weaken",
            "delta": 0.2
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["details"]["operation"] == "weaken"
    assert data["details"]["initial_weight"] == 0.5
    assert data["details"]["final_weight"] == 0.3
    print(f"✓ Weakened synapse: {data['message']}")


def test_set_synapse_weight():
    """Test setting synapse weight directly"""
    # Create two neurons and a synapse
    neuron1_id = create_test_neuron()
    neuron2_id = create_test_neuron()
    synapse_id = create_test_synapse(neuron1_id, neuron2_id)
    
    # Set weight directly
    response = client.post(
        "/api/v1/training/adjust-synapse",
        json={
            "synapse_id": synapse_id,
            "operation": "set",
            "new_weight": 0.9
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["details"]["operation"] == "set"
    assert data["details"]["final_weight"] == 0.9
    print(f"✓ Set synapse weight: {data['message']}")


def test_synapse_deletion_on_zero_weight():
    """Test that synapse is deleted when weight reaches zero"""
    # Create two neurons and a synapse with low weight
    neuron1_id = create_test_neuron()
    neuron2_id = create_test_neuron()
    synapse_id = create_test_synapse(neuron1_id, neuron2_id)
    
    # Set weight to near zero
    response = client.post(
        "/api/v1/training/adjust-synapse",
        json={
            "synapse_id": synapse_id,
            "operation": "set",
            "new_weight": 0.005
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["details"]["deleted"] is True
    print(f"✓ Synapse deleted when weight reached zero: {data['message']}")


def test_create_tool_neuron():
    """Test creating a tool neuron"""
    response = client.post(
        "/api/v1/training/create-tool",
        json={
            "description": "Calculate the sum of two numbers",
            "function_signature": "add(x: float, y: float)",
            "executable_code": "result = x + y",
            "input_schema": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"}
                },
                "required": ["x", "y"]
            },
            "output_schema": {
                "type": "number"
            }
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "tool_neuron_id" in data["details"]
    assert data["details"]["auto_connected"] is True
    print(f"✓ Created tool neuron: {data['details']['tool_neuron_id']}")


def test_create_tool_neuron_with_manual_connections():
    """Test creating a tool neuron with manual connections"""
    # Create a knowledge neuron to connect to
    knowledge_neuron_id = create_test_neuron()
    
    response = client.post(
        "/api/v1/training/create-tool",
        json={
            "description": "Process text data",
            "function_signature": "process_text(text: str)",
            "executable_code": "result = text.upper()",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                }
            },
            "output_schema": {
                "type": "string"
            },
            "connect_to_neurons": [knowledge_neuron_id]
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["details"]["manual_connections"] == 1
    print(f"✓ Created tool neuron with manual connections")


def test_create_tool_neuron_invalid_code():
    """Test creating tool neuron with invalid code"""
    response = client.post(
        "/api/v1/training/create-tool",
        json={
            "description": "Invalid tool",
            "function_signature": "invalid()",
            "executable_code": "import os; os.system('ls')",  # Dangerous code
            "input_schema": {},
            "output_schema": {}
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    
    assert response.status_code == 400  # Should be rejected
    print("✓ Correctly rejected dangerous code")


def test_adjust_nonexistent_neuron():
    """Test adjusting a neuron that doesn't exist"""
    response = client.post(
        "/api/v1/training/adjust-neuron",
        json={
            "neuron_id": "00000000-0000-0000-0000-000000000000",
            "target_text": "Test",
            "learning_rate": 0.1
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    
    assert response.status_code == 404
    print("✓ Correctly handled nonexistent neuron")


def test_adjust_nonexistent_synapse():
    """Test adjusting a synapse that doesn't exist"""
    response = client.post(
        "/api/v1/training/adjust-synapse",
        json={
            "synapse_id": "00000000-0000-0000-0000-000000000000",
            "operation": "strengthen",
            "delta": 0.1
        },
        headers={"X-API-Key": "dev-key-12345"}
    )
    
    assert response.status_code == 404
    print("✓ Correctly handled nonexistent synapse")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
