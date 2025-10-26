"""
Example usage of Tool Clusters for complex functionality.

This example demonstrates how to create and execute tool clusters
that orchestrate multiple tool neurons in a directed acyclic graph (DAG).
"""

from neuron_system.core.graph import NeuronGraph
from neuron_system.core.vector3d import Vector3D
from neuron_system.neuron_types.tool_neuron import ToolNeuron
from neuron_system.tools.tool_cluster import ToolCluster
from neuron_system.tools.tool_executor import ToolClusterExecutor


def example_data_processing_pipeline():
    """
    Example: Create a data processing pipeline cluster.
    
    Pipeline: Load Data -> Clean Data -> Transform Data -> Aggregate Results
    """
    print("=" * 70)
    print("Example: Data Processing Pipeline Cluster")
    print("=" * 70)
    
    # Create graph
    graph = NeuronGraph()
    
    # 1. Load Data Tool
    load_tool = ToolNeuron(
        function_signature="load_data(source)",
        executable_code="""
# Simulate loading data
source = inputs.get('source', 'default')
result = {
    'data': [1, 2, 3, 4, 5, None, 6, 7, 8, None, 9, 10],
    'source': source,
    'count': 12
}
""",
        activation_threshold=0.0
    )
    load_tool.position = Vector3D(0, 0, 0)
    load_tool.metadata['description'] = "Loads raw data from source"
    
    # 2. Clean Data Tool
    clean_tool = ToolNeuron(
        function_signature="clean_data(data)",
        executable_code="""
# Get data from previous tool
import uuid
load_tool_id = inputs.get('load_tool_id')
data_key = f'dep_{load_tool_id}'
data_dict = inputs.get(data_key, {})
raw_data = data_dict.get('data', [])

# Clean: remove None values
cleaned = [x for x in raw_data if x is not None]
result = {
    'data': cleaned,
    'removed': len(raw_data) - len(cleaned),
    'count': len(cleaned)
}
""",
        activation_threshold=0.0
    )
    clean_tool.position = Vector3D(5, 0, 0)
    clean_tool.metadata['description'] = "Removes invalid data points"
    
    # 3. Transform Data Tool
    transform_tool = ToolNeuron(
        function_signature="transform_data(data, operation)",
        executable_code="""
# Get cleaned data
import uuid
clean_tool_id = inputs.get('clean_tool_id')
data_key = f'dep_{clean_tool_id}'
data_dict = inputs.get(data_key, {})
data = data_dict.get('data', [])

# Transform: square each value
operation = inputs.get('operation', 'square')
if operation == 'square':
    transformed = [x * x for x in data]
elif operation == 'double':
    transformed = [x * 2 for x in data]
else:
    transformed = data

result = {
    'data': transformed,
    'operation': operation,
    'count': len(transformed)
}
""",
        activation_threshold=0.0
    )
    transform_tool.position = Vector3D(10, 0, 0)
    transform_tool.metadata['description'] = "Applies transformation to data"
    
    # 4. Aggregate Results Tool
    aggregate_tool = ToolNeuron(
        function_signature="aggregate(data)",
        executable_code="""
# Get transformed data
import uuid
transform_tool_id = inputs.get('transform_tool_id')
data_key = f'dep_{transform_tool_id}'
data_dict = inputs.get(data_key, {})
data = data_dict.get('data', [])

# Aggregate
result = {
    'sum': sum(data),
    'average': sum(data) / len(data) if data else 0,
    'min': min(data) if data else 0,
    'max': max(data) if data else 0,
    'count': len(data)
}
""",
        activation_threshold=0.0
    )
    aggregate_tool.position = Vector3D(15, 0, 0)
    aggregate_tool.metadata['description'] = "Computes aggregate statistics"
    
    # Add tools to graph
    graph.add_neuron(load_tool)
    graph.add_neuron(clean_tool)
    graph.add_neuron(transform_tool)
    graph.add_neuron(aggregate_tool)
    
    print(f"\n✓ Created 4 tool neurons")
    
    # Create cluster with execution graph
    cluster = ToolCluster(
        name="data_processing_pipeline",
        input_interface={
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "operation": {"type": "string"}
            }
        },
        output_interface={
            "type": "object",
            "properties": {
                "sum": {"type": "number"},
                "average": {"type": "number"},
                "min": {"type": "number"},
                "max": {"type": "number"},
                "count": {"type": "integer"}
            }
        },
        metadata={
            "description": "Complete data processing pipeline",
            "capabilities": ["data-processing", "etl", "analytics"],
            "version": "1.0.0"
        }
    )
    
    # Add tools to cluster
    cluster.add_tool_neuron(load_tool.id)
    cluster.add_tool_neuron(clean_tool.id)
    cluster.add_tool_neuron(transform_tool.id)
    cluster.add_tool_neuron(aggregate_tool.id)
    
    # Define execution dependencies (DAG)
    cluster.add_dependency(clean_tool.id, load_tool.id)      # clean depends on load
    cluster.add_dependency(transform_tool.id, clean_tool.id) # transform depends on clean
    cluster.add_dependency(aggregate_tool.id, transform_tool.id) # aggregate depends on transform
    
    print(f"✓ Created cluster with 4 tools and 3 dependencies")
    
    # Validate cluster
    assert cluster.validate_acyclic(), "Cluster has cycles!"
    print(f"✓ Cluster is acyclic (valid DAG)")
    
    # Get execution order
    order = cluster.get_execution_order()
    print(f"\n📋 Execution order:")
    for i, tool_id in enumerate(order, 1):
        tool = graph.get_neuron(tool_id)
        print(f"   {i}. {tool.function_signature} - {tool.metadata.get('description')}")
    
    # Add cluster to graph
    graph.add_cluster(cluster)
    print(f"\n✓ Cluster added to graph")
    
    # Execute the cluster
    print(f"\n🚀 Executing cluster...")
    result = graph.execute_cluster_by_name(
        "data_processing_pipeline",
        {
            "source": "sensor_data.csv",
            "operation": "square",
            "load_tool_id": str(load_tool.id),
            "clean_tool_id": str(clean_tool.id),
            "transform_tool_id": str(transform_tool.id)
        }
    )
    
    print(f"\n📊 Execution Results:")
    print(f"   Status: {result['status']}")
    print(f"   Tools executed: {result['tools_executed']}/{len(cluster.tool_neurons)}")
    print(f"   Execution time: {result['execution_time_ms']:.2f}ms")
    
    if result['results']:
        print(f"\n📈 Final Results:")
        for key, value in result['results'].items():
            if not key.startswith('output_'):
                print(f"   {key}: {value}")
    
    print(f"\n✓ Pipeline completed successfully!")
    
    return graph, cluster


def example_parallel_processing():
    """
    Example: Create a cluster with parallel execution branches.
    
    Structure:
              -> Process A -> 
    Input ->  -> Process B ->  -> Combine -> Output
              -> Process C ->
    """
    print("\n" + "=" * 70)
    print("Example: Parallel Processing Cluster")
    print("=" * 70)
    
    graph = NeuronGraph()
    
    # Input tool
    input_tool = ToolNeuron(
        function_signature="prepare_input(value)",
        executable_code="result = {'value': inputs.get('value', 0), 'timestamp': 'now'}",
        activation_threshold=0.0
    )
    input_tool.position = Vector3D(0, 0, 0)
    
    # Parallel processing tools
    process_a = ToolNeuron(
        function_signature="process_a(data)",
        executable_code="result = {'result_a': inputs.get('value', 0) * 2}",
        activation_threshold=0.0
    )
    process_a.position = Vector3D(5, 5, 0)
    
    process_b = ToolNeuron(
        function_signature="process_b(data)",
        executable_code="result = {'result_b': inputs.get('value', 0) ** 2}",
        activation_threshold=0.0
    )
    process_b.position = Vector3D(5, 0, 0)
    
    process_c = ToolNeuron(
        function_signature="process_c(data)",
        executable_code="result = {'result_c': inputs.get('value', 0) + 10}",
        activation_threshold=0.0
    )
    process_c.position = Vector3D(5, -5, 0)
    
    # Combine tool
    combine_tool = ToolNeuron(
        function_signature="combine(results)",
        executable_code="""
# Combine all results
result = {
    'combined': inputs.get('result_a', 0) + inputs.get('result_b', 0) + inputs.get('result_c', 0),
    'details': {
        'a': inputs.get('result_a', 0),
        'b': inputs.get('result_b', 0),
        'c': inputs.get('result_c', 0)
    }
}
""",
        activation_threshold=0.0
    )
    combine_tool.position = Vector3D(10, 0, 0)
    
    # Add to graph
    for tool in [input_tool, process_a, process_b, process_c, combine_tool]:
        graph.add_neuron(tool)
    
    # Create cluster
    cluster = ToolCluster(name="parallel_processor")
    cluster.add_tool_neuron(input_tool.id)
    cluster.add_tool_neuron(process_a.id)
    cluster.add_tool_neuron(process_b.id)
    cluster.add_tool_neuron(process_c.id)
    cluster.add_tool_neuron(combine_tool.id)
    
    # Define parallel structure
    cluster.add_dependency(process_a.id, input_tool.id)
    cluster.add_dependency(process_b.id, input_tool.id)
    cluster.add_dependency(process_c.id, input_tool.id)
    cluster.add_dependency(combine_tool.id, process_a.id)
    cluster.add_dependency(combine_tool.id, process_b.id)
    cluster.add_dependency(combine_tool.id, process_c.id)
    
    print(f"\n✓ Created parallel processing cluster")
    print(f"   Input tools: {len(cluster.get_input_tools())}")
    print(f"   Output tools: {len(cluster.get_output_tools())}")
    
    # Validate
    executor = ToolClusterExecutor(graph)
    validation = executor.validate_cluster(cluster)
    print(f"\n✓ Cluster validation: {validation['valid']}")
    
    graph.add_cluster(cluster)
    
    # Execute
    result = graph.execute_cluster_by_name(
        "parallel_processor",
        {"value": 5}
    )
    
    print(f"\n📊 Execution Results:")
    print(f"   Status: {result['status']}")
    print(f"   Execution time: {result['execution_time_ms']:.2f}ms")
    print(f"   Final result: {result['results']}")
    
    return graph, cluster


def example_cluster_management():
    """
    Example: Demonstrate cluster management operations.
    """
    print("\n" + "=" * 70)
    print("Example: Cluster Management")
    print("=" * 70)
    
    graph = NeuronGraph()
    
    # Create multiple clusters
    for i in range(3):
        tool = ToolNeuron(
            function_signature=f"tool_{i}()",
            executable_code=f"result = {i}",
            activation_threshold=0.0
        )
        tool.position = Vector3D(i * 5, 0, 0)
        graph.add_neuron(tool)
        
        cluster = ToolCluster(
            name=f"cluster_{i}",
            metadata={
                "capabilities": ["processing", f"type_{i}"],
                "version": f"1.{i}.0"
            }
        )
        cluster.add_tool_neuron(tool.id)
        graph.add_cluster(cluster)
    
    print(f"\n✓ Created {graph.get_cluster_count()} clusters")
    
    # List all clusters
    print(f"\n📋 All clusters:")
    for cluster in graph.list_clusters():
        print(f"   - {cluster.name} (ID: {str(cluster.id)[:8]}...)")
    
    # Query by name
    cluster = graph.get_cluster_by_name("cluster_1")
    print(f"\n✓ Retrieved cluster by name: {cluster.name}")
    
    # Query by capability
    results = graph.query_clusters_by_capability("processing")
    print(f"\n✓ Found {len(results)} clusters with 'processing' capability")
    
    # Remove a cluster
    removed = graph.remove_cluster(cluster.id)
    print(f"\n✓ Removed cluster: {removed}")
    print(f"   Remaining clusters: {graph.get_cluster_count()}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Tool Cluster Usage Examples")
    print("=" * 70)
    
    # Run examples
    example_data_processing_pipeline()
    example_parallel_processing()
    example_cluster_management()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully! ✓")
    print("=" * 70)
