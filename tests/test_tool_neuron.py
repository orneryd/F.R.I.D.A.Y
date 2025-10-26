"""
Test Tool Neuron functionality.
"""

import numpy as np
from uuid import uuid4

from neuron_system.core.graph import NeuronGraph
from neuron_system.core.vector3d import Vector3D
from neuron_system.core.synapse import Synapse, SynapseType
from neuron_system.neuron_types.tool_neuron import ToolNeuron
from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
from neuron_system.engines.training import TrainingEngine
from neuron_system.engines.query import QueryEngine


def test_tool_neuron_creation():
    """Test creating a tool neuron with execution capability."""
    print("\n=== Test: Tool Neuron Creation ===")
    
    # Create a simple tool neuron
    tool = ToolNeuron(
        function_signature="add(a: number, b: number)",
        executable_code="result = inputs.get('a', 0) + inputs.get('b', 0)",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        },
        output_schema={
            "type": "number"
        }
    )
    
    tool.id = uuid4()
    tool.position = Vector3D(0, 0, 0)
    tool.vector = np.random.randn(384)
    
    # Test execution
    result = tool.execute({"a": 5, "b": 3})
    print(f"Tool execution result: {result}")
    assert result == 8, f"Expected 8, got {result}"
    
    # Check execution stats
    print(f"Execution count: {tool.execution_count}")
    print(f"Average execution time: {tool.average_execution_time:.2f}ms")
    
    print("✓ Tool neuron creation and execution successful")


def test_tool_neuron_input_extraction():
    """Test extracting inputs from connected neurons."""
    print("\n=== Test: Tool Input Extraction ===")
    
    # Create graph
    graph = NeuronGraph()
    
    # Create knowledge neuron with data
    knowledge = KnowledgeNeuron(
        source_data="The answer is 42",
        compression_ratio=10.0
    )
    knowledge.id = uuid4()
    knowledge.position = Vector3D(0, 0, 0)
    knowledge.vector = np.random.randn(384)
    graph.add_neuron(knowledge)
    
    # Create tool neuron
    tool = ToolNeuron(
        function_signature="process(data: string)",
        executable_code="result = f'Processed: {inputs.get(\"data\", \"\")}'",
        input_schema={
            "type": "object",
            "properties": {
                "data": {"type": "string"}
            }
        }
    )
    tool.id = uuid4()
    tool.position = Vector3D(1, 1, 1)
    tool.vector = np.random.randn(384)
    graph.add_neuron(tool)
    
    # Connect with TOOL_INPUT synapse
    synapse = Synapse(
        id=uuid4(),
        source_neuron_id=knowledge.id,
        target_neuron_id=tool.id,
        weight=0.8,
        synapse_type=SynapseType.TOOL_INPUT,
        metadata={"parameter_name": "data"}
    )
    graph.add_synapse(synapse)
    
    # Extract inputs
    activated_neurons = {knowledge.id: 0.9}
    inputs = tool.extract_inputs_from_synapses(graph, activated_neurons)
    
    print(f"Extracted inputs: {inputs}")
    assert "data" in inputs, "Expected 'data' parameter in inputs"
    assert inputs["data"] == "The answer is 42", f"Expected knowledge data, got {inputs['data']}"
    
    print("✓ Tool input extraction successful")


def test_tool_neuron_output_propagation():
    """Test propagating results to output synapses."""
    print("\n=== Test: Tool Output Propagation ===")
    
    # Create graph
    graph = NeuronGraph()
    
    # Create tool neuron
    tool = ToolNeuron(
        function_signature="calculate()",
        executable_code="result = 100"
    )
    tool.id = uuid4()
    tool.position = Vector3D(0, 0, 0)
    tool.vector = np.random.randn(384)
    graph.add_neuron(tool)
    
    # Create target knowledge neuron
    target = KnowledgeNeuron(
        source_data="Target neuron",
        compression_ratio=5.0
    )
    target.id = uuid4()
    target.position = Vector3D(1, 1, 1)
    target.vector = np.random.randn(384)
    graph.add_neuron(target)
    
    # Connect with TOOL_OUTPUT synapse
    synapse = Synapse(
        id=uuid4(),
        source_neuron_id=tool.id,
        target_neuron_id=target.id,
        weight=0.9,
        synapse_type=SynapseType.TOOL_OUTPUT
    )
    graph.add_synapse(synapse)
    
    # Propagate result
    result = 100
    propagated_to = tool.propagate_results_to_outputs(graph, result)
    
    print(f"Propagated to {len(propagated_to)} neurons")
    assert len(propagated_to) == 1, f"Expected 1 neuron, got {len(propagated_to)}"
    assert target.id in propagated_to, "Expected target neuron in propagated list"
    
    # Check target received the result
    assert "received_tool_result" in target.metadata, "Target should have received result"
    received = target.metadata["received_tool_result"]
    print(f"Target received: {received['result']}")
    assert received["result"] == 100, f"Expected 100, got {received['result']}"
    
    print("✓ Tool output propagation successful")


def test_dynamic_tool_creation():
    """Test creating tool neurons dynamically via TrainingEngine."""
    print("\n=== Test: Dynamic Tool Creation ===")
    
    # Create graph and training engine
    graph = NeuronGraph()
    training_engine = TrainingEngine(graph)
    
    # Add some knowledge neurons for auto-connection
    for i in range(3):
        knowledge = KnowledgeNeuron(
            source_data=f"Knowledge item {i}",
            compression_ratio=10.0
        )
        knowledge.id = uuid4()
        knowledge.position = Vector3D(i, i, i)
        knowledge.vector = np.random.randn(384)
        graph.add_neuron(knowledge)
    
    # Create a tool neuron dynamically
    description = "Calculate the sum of two numbers"
    code = "result = inputs.get('x', 0) + inputs.get('y', 0)"
    input_schema = {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"}
        },
        "required": ["x", "y"]
    }
    
    tool_id = training_engine.create_tool_neuron(
        description=description,
        code=code,
        input_schema=input_schema,
        auto_connect=True,
        connection_threshold=0.5
    )
    
    print(f"Created tool neuron: {tool_id}")
    
    # Verify the tool was created
    tool = graph.get_neuron(tool_id)
    assert tool is not None, "Tool neuron should exist in graph"
    assert tool.executable_code == code, "Code should match"
    
    # Check if it was auto-connected
    incoming = graph.get_incoming_synapses(tool_id)
    print(f"Auto-connected to {len(incoming)} neurons")
    
    print("✓ Dynamic tool creation successful")


def test_tool_execution_in_query():
    """Test tool execution during query processing."""
    print("\n=== Test: Tool Execution in Query ===")
    
    # Create graph
    graph = NeuronGraph()
    
    # Create a simple tool neuron
    tool = ToolNeuron(
        function_signature="greet(name: string)",
        executable_code="result = f'Hello, {inputs.get(\"name\", \"World\")}!'",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        },
        activation_threshold=0.3
    )
    tool.id = uuid4()
    tool.position = Vector3D(0, 0, 0)
    tool.vector = np.random.randn(384)
    graph.add_neuron(tool)
    
    # Create knowledge neuron with name
    knowledge = KnowledgeNeuron(
        source_data="Alice",
        compression_ratio=5.0
    )
    knowledge.id = uuid4()
    knowledge.position = Vector3D(1, 0, 0)
    knowledge.vector = np.random.randn(384)
    graph.add_neuron(knowledge)
    
    # Connect them
    synapse = Synapse(
        id=uuid4(),
        source_neuron_id=knowledge.id,
        target_neuron_id=tool.id,
        weight=0.8,
        synapse_type=SynapseType.TOOL_INPUT,
        metadata={"parameter_name": "name"}
    )
    graph.add_synapse(synapse)
    
    # Create query engine
    query_engine = QueryEngine(graph)
    
    # Create activated neurons list
    from neuron_system.engines.query import ActivatedNeuron
    activated_neurons = [
        ActivatedNeuron(neuron=knowledge, activation=0.9),
        ActivatedNeuron(neuron=tool, activation=0.7)
    ]
    
    # Execute tool neurons
    tool_results = query_engine.execute_tool_neurons(
        activated_neurons,
        execution_threshold=0.5
    )
    
    print(f"Tool execution results: {tool_results}")
    assert tool_results["total_executed"] == 1, f"Expected 1 execution, got {tool_results['total_executed']}"
    assert tool_results["total_failed"] == 0, f"Expected 0 failures, got {tool_results['total_failed']}"
    
    # Check the result
    executed = tool_results["executed_tools"][0]
    print(f"Tool result: {executed['result']}")
    assert "Hello" in str(executed["result"]), "Expected greeting in result"
    
    print("✓ Tool execution in query successful")


if __name__ == "__main__":
    print("Testing Tool Neuron Functionality")
    print("=" * 50)
    
    try:
        test_tool_neuron_creation()
        test_tool_neuron_input_extraction()
        test_tool_neuron_output_propagation()
        test_dynamic_tool_creation()
        test_tool_execution_in_query()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
