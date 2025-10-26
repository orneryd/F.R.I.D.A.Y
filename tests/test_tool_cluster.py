"""
Test tool cluster functionality.
"""

from uuid import uuid4
from neuron_system.core.graph import NeuronGraph
from neuron_system.core.vector3d import Vector3D
from neuron_system.neuron_types.tool_neuron import ToolNeuron
from neuron_system.tools.tool_cluster import ToolCluster
from neuron_system.tools.tool_executor import ToolClusterExecutor


def test_tool_cluster_creation():
    """Test creating a tool cluster."""
    print("Testing tool cluster creation...")
    
    # Create a simple cluster
    cluster = ToolCluster(
        name="test_cluster",
        input_interface={"type": "object"},
        output_interface={"type": "object"}
    )
    
    assert cluster.name == "test_cluster"
    assert cluster.id is not None
    assert len(cluster.tool_neurons) == 0
    print("✓ Tool cluster created successfully")


def test_tool_cluster_dag():
    """Test DAG validation and execution order."""
    print("\nTesting DAG validation and execution order...")
    
    # Create tool neurons
    tool1_id = uuid4()
    tool2_id = uuid4()
    tool3_id = uuid4()
    
    # Create cluster with dependencies: tool3 -> tool2 -> tool1
    cluster = ToolCluster(name="dag_test")
    cluster.add_tool_neuron(tool1_id)
    cluster.add_tool_neuron(tool2_id)
    cluster.add_tool_neuron(tool3_id)
    
    # Add dependencies
    cluster.add_dependency(tool2_id, tool1_id)  # tool2 depends on tool1
    cluster.add_dependency(tool3_id, tool2_id)  # tool3 depends on tool2
    
    # Validate acyclic
    assert cluster.validate_acyclic() == True
    print("✓ DAG is acyclic")
    
    # Get execution order
    order = cluster.get_execution_order()
    assert len(order) == 3
    assert order.index(tool1_id) < order.index(tool2_id)
    assert order.index(tool2_id) < order.index(tool3_id)
    print(f"✓ Execution order correct: {[str(t)[:8] for t in order]}")
    
    # Test cycle detection
    cluster.add_dependency(tool1_id, tool3_id)  # Create cycle
    assert cluster.validate_acyclic() == False
    print("✓ Cycle detection works")


def test_cluster_with_graph():
    """Test cluster integration with NeuronGraph."""
    print("\nTesting cluster integration with graph...")
    
    # Create graph
    graph = NeuronGraph()
    
    # Create tool neurons
    tool1 = ToolNeuron(
        function_signature="add(a, b)",
        executable_code="result = inputs.get('a', 0) + inputs.get('b', 0)",
        input_schema={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
        output_schema={"type": "number"}
    )
    tool1.position = Vector3D(0, 0, 0)
    
    tool2 = ToolNeuron(
        function_signature="multiply(x, factor)",
        executable_code="result = inputs.get('x', 0) * inputs.get('factor', 1)",
        input_schema={"type": "object", "properties": {"x": {"type": "number"}, "factor": {"type": "number"}}},
        output_schema={"type": "number"}
    )
    tool2.position = Vector3D(1, 1, 1)
    
    # Add to graph
    graph.add_neuron(tool1)
    graph.add_neuron(tool2)
    
    # Create cluster
    cluster = ToolCluster(name="math_operations")
    cluster.add_tool_neuron(tool1.id)
    cluster.add_tool_neuron(tool2.id)
    cluster.add_dependency(tool2.id, tool1.id)  # multiply depends on add
    
    # Add cluster to graph
    cluster_id = graph.add_cluster(cluster)
    assert cluster_id == cluster.id
    print("✓ Cluster added to graph")
    
    # Retrieve cluster
    retrieved = graph.get_cluster(cluster_id)
    assert retrieved is not None
    assert retrieved.name == "math_operations"
    print("✓ Cluster retrieved by ID")
    
    # Retrieve by name
    retrieved_by_name = graph.get_cluster_by_name("math_operations")
    assert retrieved_by_name is not None
    assert retrieved_by_name.id == cluster_id
    print("✓ Cluster retrieved by name")
    
    # Query by capability
    cluster.metadata["capabilities"] = ["math", "arithmetic"]
    results = graph.query_clusters_by_capability("math")
    assert len(results) == 1
    assert results[0].id == cluster_id
    print("✓ Cluster queried by capability")


def test_cluster_execution():
    """Test executing a tool cluster."""
    print("\nTesting cluster execution...")
    
    # Create graph
    graph = NeuronGraph()
    
    # Create tool neurons
    tool1 = ToolNeuron(
        function_signature="add(a, b)",
        executable_code="result = inputs.get('a', 0) + inputs.get('b', 0)",
        activation_threshold=0.0
    )
    tool1.position = Vector3D(0, 0, 0)
    
    tool2 = ToolNeuron(
        function_signature="multiply(x, factor)",
        executable_code="result = inputs.get('dep_' + str(inputs.get('tool1_id')), 0) * inputs.get('factor', 2)",
        activation_threshold=0.0
    )
    tool2.position = Vector3D(1, 1, 1)
    
    graph.add_neuron(tool1)
    graph.add_neuron(tool2)
    
    # Create cluster
    cluster = ToolCluster(name="math_pipeline")
    cluster.add_tool_neuron(tool1.id)
    cluster.add_tool_neuron(tool2.id)
    cluster.add_dependency(tool2.id, tool1.id)
    
    graph.add_cluster(cluster)
    
    # Execute cluster
    result = graph.execute_cluster_by_name(
        "math_pipeline",
        {"a": 5, "b": 3, "factor": 2, "tool1_id": str(tool1.id)}
    )
    
    assert result["status"] in ["success", "partial_success"]
    assert result["cluster_name"] == "math_pipeline"
    assert "results" in result
    print(f"✓ Cluster executed: {result['status']}")
    print(f"  Tools executed: {result['tools_executed']}")
    print(f"  Execution time: {result['execution_time_ms']:.2f}ms")


def test_cluster_validation():
    """Test cluster validation."""
    print("\nTesting cluster validation...")
    
    graph = NeuronGraph()
    
    # Create tool neuron
    tool1 = ToolNeuron(
        function_signature="test()",
        executable_code="result = 42"
    )
    tool1.position = Vector3D(0, 0, 0)
    graph.add_neuron(tool1)
    
    # Create cluster
    cluster = ToolCluster(name="test_cluster")
    cluster.add_tool_neuron(tool1.id)
    graph.add_cluster(cluster)
    
    # Validate
    executor = ToolClusterExecutor(graph)
    validation = executor.validate_cluster(cluster)
    
    assert validation["valid"] == True
    assert validation["tool_count"] == 1
    print("✓ Cluster validation passed")
    print(f"  Tool count: {validation['tool_count']}")
    print(f"  Input tools: {validation['input_tools']}")
    print(f"  Output tools: {validation['output_tools']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Tool Cluster Tests")
    print("=" * 60)
    
    test_tool_cluster_creation()
    test_tool_cluster_dag()
    test_cluster_with_graph()
    test_cluster_execution()
    test_cluster_validation()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
