# Tool Cluster Implementation

## Overview

Tool Clusters enable grouping multiple Tool Neurons into complex, orchestrated workflows. Clusters execute tools in a directed acyclic graph (DAG) structure, allowing for sophisticated data processing pipelines, parallel execution, and complex functionality composition.

## Key Features

### 1. DAG-Based Execution
- Tools are organized in a directed acyclic graph (DAG)
- Automatic cycle detection prevents invalid configurations
- Topological sort determines optimal execution order
- Dependencies ensure proper data flow between tools

### 2. Flexible Orchestration
- Sequential pipelines (A → B → C)
- Parallel branches (A → [B, C, D] → E)
- Complex workflows with multiple dependencies
- Automatic data flow between dependent tools

### 3. Cluster Management
- Store clusters in the neuron graph
- Query clusters by name or capability
- Execute clusters as callable units
- Track execution statistics and performance

## Architecture

### Core Components

```
ToolCluster
├── Tool Neurons (List of UUIDs)
├── Execution Graph (DAG of dependencies)
├── Input Interface (Schema)
├── Output Interface (Schema)
└── Metadata (Capabilities, version, etc.)

ToolClusterExecutor
├── Validates DAG structure
├── Computes execution order
├── Orchestrates tool execution
├── Manages data flow
└── Aggregates results

NeuronGraph Extensions
├── add_cluster()
├── remove_cluster()
├── get_cluster() / get_cluster_by_name()
├── query_clusters_by_capability()
└── execute_cluster() / execute_cluster_by_name()
```

## Usage Examples

### Example 1: Sequential Pipeline

```python
from neuron_system.core.graph import NeuronGraph
from neuron_system.core.vector3d import Vector3D
from neuron_system.neuron_types.tool_neuron import ToolNeuron
from neuron_system.tools.tool_cluster import ToolCluster

# Create graph
graph = NeuronGraph()

# Create tools
load_tool = ToolNeuron(
    function_signature="load_data(source)",
    executable_code="result = {'data': [1, 2, 3, 4, 5]}",
    activation_threshold=0.0
)
load_tool.position = Vector3D(0, 0, 0)

process_tool = ToolNeuron(
    function_signature="process_data(data)",
    executable_code="result = {'processed': [x * 2 for x in inputs.get('data', [])]}",
    activation_threshold=0.0
)
process_tool.position = Vector3D(5, 0, 0)

# Add to graph
graph.add_neuron(load_tool)
graph.add_neuron(process_tool)

# Create cluster
cluster = ToolCluster(name="data_pipeline")
cluster.add_tool_neuron(load_tool.id)
cluster.add_tool_neuron(process_tool.id)
cluster.add_dependency(process_tool.id, load_tool.id)

# Add cluster to graph
graph.add_cluster(cluster)

# Execute
result = graph.execute_cluster_by_name("data_pipeline", {})
print(result['results'])
```

### Example 2: Parallel Processing

```python
# Create input tool
input_tool = ToolNeuron(
    function_signature="prepare(value)",
    executable_code="result = {'value': inputs.get('value', 0)}",
    activation_threshold=0.0
)
input_tool.position = Vector3D(0, 0, 0)

# Create parallel processing tools
process_a = ToolNeuron(
    function_signature="process_a(data)",
    executable_code="result = {'a': inputs.get('value', 0) * 2}",
    activation_threshold=0.0
)
process_a.position = Vector3D(5, 5, 0)

process_b = ToolNeuron(
    function_signature="process_b(data)",
    executable_code="result = {'b': inputs.get('value', 0) ** 2}",
    activation_threshold=0.0
)
process_b.position = Vector3D(5, -5, 0)

# Create combine tool
combine_tool = ToolNeuron(
    function_signature="combine(results)",
    executable_code="result = {'sum': inputs.get('a', 0) + inputs.get('b', 0)}",
    activation_threshold=0.0
)
combine_tool.position = Vector3D(10, 0, 0)

# Add to graph
for tool in [input_tool, process_a, process_b, combine_tool]:
    graph.add_neuron(tool)

# Create cluster with parallel structure
cluster = ToolCluster(name="parallel_processor")
cluster.add_tool_neuron(input_tool.id)
cluster.add_tool_neuron(process_a.id)
cluster.add_tool_neuron(process_b.id)
cluster.add_tool_neuron(combine_tool.id)

# Define parallel dependencies
cluster.add_dependency(process_a.id, input_tool.id)
cluster.add_dependency(process_b.id, input_tool.id)
cluster.add_dependency(combine_tool.id, process_a.id)
cluster.add_dependency(combine_tool.id, process_b.id)

graph.add_cluster(cluster)

# Execute
result = graph.execute_cluster_by_name("parallel_processor", {"value": 5})
# Result: {'sum': 35} (5*2 + 5^2 = 10 + 25 = 35)
```

### Example 3: Cluster Management

```python
# Query clusters by capability
cluster.metadata["capabilities"] = ["data-processing", "etl"]
results = graph.query_clusters_by_capability("etl")

# Get cluster by name
cluster = graph.get_cluster_by_name("data_pipeline")

# List all clusters
all_clusters = graph.list_clusters()

# Remove cluster
graph.remove_cluster(cluster.id)
```

## API Reference

### ToolCluster

#### Constructor
```python
ToolCluster(
    name: str,
    tool_neurons: List[UUID] = None,
    execution_graph: Dict[UUID, List[UUID]] = None,
    input_interface: Dict[str, Any] = None,
    output_interface: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None
)
```

#### Methods

**add_tool_neuron(tool_neuron_id: UUID)**
- Adds a tool neuron to the cluster

**remove_tool_neuron(tool_neuron_id: UUID) -> bool**
- Removes a tool neuron from the cluster

**add_dependency(tool_id: UUID, depends_on: UUID)**
- Adds a dependency: tool_id depends on depends_on

**remove_dependency(tool_id: UUID, depends_on: UUID) -> bool**
- Removes a dependency

**validate_acyclic() -> bool**
- Validates that the execution graph is acyclic (no cycles)

**get_execution_order() -> List[UUID]**
- Computes topological sort for execution order

**get_input_tools() -> List[UUID]**
- Returns tools with no dependencies (entry points)

**get_output_tools() -> List[UUID]**
- Returns tools with no dependents (exit points)

**to_dict() -> Dict[str, Any]**
- Serializes cluster to dictionary

**from_dict(data: Dict[str, Any]) -> ToolCluster**
- Deserializes cluster from dictionary

### ToolClusterExecutor

#### Constructor
```python
ToolClusterExecutor(graph: NeuronGraph)
```

#### Methods

**execute_cluster(cluster: ToolCluster, inputs: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]**
- Executes all tools in the cluster according to the DAG
- Returns execution results with status, timing, and errors

**validate_cluster(cluster: ToolCluster) -> Dict[str, Any]**
- Validates cluster structure and dependencies
- Returns validation results with errors and warnings

### NeuronGraph Extensions

**add_cluster(cluster: ToolCluster) -> UUID**
- Adds a tool cluster to the graph

**remove_cluster(cluster_id: UUID) -> bool**
- Removes a tool cluster from the graph

**get_cluster(cluster_id: UUID) -> Optional[ToolCluster]**
- Retrieves a cluster by ID

**get_cluster_by_name(name: str) -> Optional[ToolCluster]**
- Retrieves a cluster by name

**query_clusters_by_capability(capability: str) -> List[ToolCluster]**
- Queries clusters by capability keyword

**execute_cluster(cluster_id: UUID, inputs: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]**
- Executes a cluster by ID

**execute_cluster_by_name(name: str, inputs: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]**
- Executes a cluster by name

**get_cluster_count() -> int**
- Returns total number of clusters

**list_clusters() -> List[ToolCluster]**
- Returns all clusters in the graph

## Execution Flow

1. **Validation**
   - Verify all tool neurons exist in graph
   - Validate execution graph is acyclic
   - Check for unreachable tools

2. **Execution Order**
   - Compute topological sort of DAG
   - Determine execution sequence

3. **Tool Execution**
   - Execute tools in computed order
   - Prepare inputs from dependencies
   - Handle errors gracefully
   - Track execution statistics

4. **Result Aggregation**
   - Collect results from output tools
   - Aggregate into final result
   - Include execution metadata

## Error Handling

### Error Policies

Clusters support configurable error handling via metadata:

```python
cluster.metadata["error_policy"] = "continue"  # Default: continue on errors
cluster.metadata["error_policy"] = "abort_on_any"  # Abort on any error
cluster.metadata["error_policy"] = "abort_on_critical"  # Abort only on critical tools

# Mark critical tools
cluster.metadata["critical_tools"] = [tool1.id, tool2.id]
```

### Execution Results

```python
{
    "cluster_id": "...",
    "cluster_name": "...",
    "status": "success" | "partial_success" | "failed",
    "results": {...},  # Aggregated results from output tools
    "tool_results": {...},  # Individual tool results
    "errors": {...},  # Errors by tool ID
    "execution_log": [...],  # Detailed execution log
    "execution_time_ms": 123.45,
    "tools_executed": 5,
    "tools_failed": 0,
    "timestamp": "..."
}
```

## Performance Considerations

### Optimization Strategies

1. **Parallel Execution** (Future Enhancement)
   - Tools with no dependencies can execute in parallel
   - Current implementation is sequential

2. **Caching**
   - Cache execution results for repeated inputs
   - Implement at cluster level

3. **Lazy Evaluation**
   - Only execute tools needed for requested outputs
   - Skip unnecessary branches

4. **Resource Management**
   - Limit concurrent tool executions
   - Implement timeouts for long-running tools

## Requirements Satisfied

This implementation satisfies the following requirements:

- **Requirement 11.1**: Tool clusters group multiple tool neurons with execution graphs
- **Requirement 11.2**: Orchestrates execution in correct order using topological sort
- **Requirement 11.3**: Supports data flow between tools via dependencies
- **Requirement 11.4**: Validates execution graphs are acyclic
- **Requirement 11.5**: Exposes clusters as callable units with defined interfaces

## Testing

Run the test suite:

```bash
python test_tool_cluster.py
```

Run usage examples:

```bash
python example_tool_cluster_usage.py
```

## Future Enhancements

1. **Parallel Execution**: Execute independent tools concurrently
2. **Conditional Branching**: Support if/else logic in execution flow
3. **Loop Support**: Allow iterative processing within clusters
4. **Cluster Composition**: Nest clusters within other clusters
5. **Visual Editor**: GUI for building and editing cluster DAGs
6. **Performance Profiling**: Detailed timing and bottleneck analysis
7. **Distributed Execution**: Execute clusters across multiple nodes

## Files

- `neuron_system/tools/tool_cluster.py` - ToolCluster class implementation
- `neuron_system/tools/tool_executor.py` - ToolClusterExecutor implementation
- `neuron_system/core/graph.py` - NeuronGraph extensions for cluster management
- `test_tool_cluster.py` - Test suite
- `example_tool_cluster_usage.py` - Usage examples
