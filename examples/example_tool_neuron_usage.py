"""
Example: Tool Neuron Functionality

This example demonstrates the complete tool neuron functionality including:
1. Creating tool neurons with execution capability
2. Connecting tools to knowledge neurons via input/output synapses
3. Executing tools during query processing
4. Dynamic tool creation via TrainingEngine
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
from neuron_system.engines.compression import CompressionEngine


def main():
    print("=" * 70)
    print("Tool Neuron Functionality Example")
    print("=" * 70)
    
    # Initialize the neuron system
    print("\n1. Initializing Neuron System...")
    graph = NeuronGraph()
    compression_engine = CompressionEngine()
    training_engine = TrainingEngine(graph)
    query_engine = QueryEngine(graph, compression_engine)
    
    # Create some knowledge neurons
    print("\n2. Creating Knowledge Neurons...")
    knowledge_data = [
        "The temperature is 72 degrees Fahrenheit",
        "The humidity is 65 percent",
        "The wind speed is 15 miles per hour",
        "Python is a programming language",
        "Machine learning uses neural networks"
    ]
    
    knowledge_neurons = []
    for i, data in enumerate(knowledge_data):
        vector, _ = compression_engine.compress(data, normalize=True)
        
        knowledge = KnowledgeNeuron(
            source_data=data,
            compression_ratio=len(data) / 384
        )
        knowledge.id = uuid4()
        knowledge.position = Vector3D(i * 10, 0, 0)
        knowledge.vector = vector
        
        graph.add_neuron(knowledge)
        knowledge_neurons.append(knowledge)
        print(f"   - Created: {data[:50]}...")
    
    # Create a temperature conversion tool manually
    print("\n3. Creating Temperature Conversion Tool (Manual)...")
    temp_tool_code = """
# Convert Fahrenheit to Celsius
fahrenheit = inputs.get('temperature', 0)
celsius = (fahrenheit - 32) * 5/9
result = {
    'fahrenheit': fahrenheit,
    'celsius': round(celsius, 2),
    'unit': 'Celsius'
}
"""
    
    temp_tool = ToolNeuron(
        function_signature="convert_temperature(temperature: number)",
        executable_code=temp_tool_code,
        input_schema={
            "type": "object",
            "properties": {
                "temperature": {"type": "number"}
            },
            "required": ["temperature"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "fahrenheit": {"type": "number"},
                "celsius": {"type": "number"},
                "unit": {"type": "string"}
            }
        },
        activation_threshold=0.4
    )
    
    temp_tool.id = uuid4()
    temp_tool.position = Vector3D(0, 10, 0)
    
    # Compress the tool description
    description = "Convert temperature from Fahrenheit to Celsius"
    vector, _ = compression_engine.compress(description, normalize=True)
    temp_tool.vector = vector
    temp_tool.metadata["description"] = description
    
    graph.add_neuron(temp_tool)
    print(f"   - Created tool: {temp_tool.function_signature}")
    
    # Connect temperature knowledge to temperature tool
    print("\n4. Connecting Knowledge to Tool (Input Synapse)...")
    temp_knowledge = knowledge_neurons[0]  # "The temperature is 72 degrees..."
    
    input_synapse = Synapse(
        id=uuid4(),
        source_neuron_id=temp_knowledge.id,
        target_neuron_id=temp_tool.id,
        weight=0.9,
        synapse_type=SynapseType.TOOL_INPUT,
        metadata={
            "parameter_name": "temperature",
            "extraction_pattern": "extract_number"
        }
    )
    graph.add_synapse(input_synapse)
    print(f"   - Connected: {temp_knowledge.source_data[:40]}... → Tool")
    
    # Create an output neuron to receive results
    print("\n5. Creating Output Neuron...")
    output_neuron = KnowledgeNeuron(
        source_data="Temperature conversion result",
        compression_ratio=5.0
    )
    output_neuron.id = uuid4()
    output_neuron.position = Vector3D(0, 20, 0)
    output_neuron.vector = np.random.randn(384)
    graph.add_neuron(output_neuron)
    
    # Connect tool output to output neuron
    output_synapse = Synapse(
        id=uuid4(),
        source_neuron_id=temp_tool.id,
        target_neuron_id=output_neuron.id,
        weight=0.95,
        synapse_type=SynapseType.TOOL_OUTPUT
    )
    graph.add_synapse(output_synapse)
    print(f"   - Connected: Tool → Output Neuron")
    
    # Execute the tool manually
    print("\n6. Executing Tool Manually...")
    activated_map = {temp_knowledge.id: 0.9}
    inputs = temp_tool.extract_inputs_from_synapses(graph, activated_map)
    print(f"   - Extracted inputs: {inputs}")
    
    # For this demo, manually set the temperature value
    inputs["temperature"] = 72
    result = temp_tool.execute(inputs)
    print(f"   - Execution result: {result}")
    
    # Propagate results
    propagated_to = temp_tool.propagate_results_to_outputs(graph, result)
    print(f"   - Propagated to {len(propagated_to)} neurons")
    
    # Create a tool dynamically using TrainingEngine
    print("\n7. Creating Tool Dynamically (via TrainingEngine)...")
    
    calculator_code = """
# Simple calculator
operation = inputs.get('operation', 'add')
a = inputs.get('a', 0)
b = inputs.get('b', 0)

if operation == 'add':
    result = a + b
elif operation == 'subtract':
    result = a - b
elif operation == 'multiply':
    result = a * b
elif operation == 'divide':
    result = a / b if b != 0 else 'Error: Division by zero'
else:
    result = 'Error: Unknown operation'
"""
    
    calculator_id = training_engine.create_tool_neuron(
        description="Perform basic arithmetic operations on two numbers",
        code=calculator_code,
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
        },
        auto_connect=True,
        connection_threshold=0.6
    )
    
    calculator_tool = graph.get_neuron(calculator_id)
    print(f"   - Created calculator tool: {calculator_id}")
    print(f"   - Function signature: {calculator_tool.function_signature}")
    
    # Test the calculator
    calc_result = calculator_tool.execute({
        "operation": "multiply",
        "a": 7,
        "b": 6
    })
    print(f"   - Test calculation (7 * 6): {calc_result}")
    
    # Execute tools during query
    print("\n8. Executing Tools During Query...")
    
    # Create activated neurons list
    from neuron_system.engines.query import ActivatedNeuron
    activated_neurons = [
        ActivatedNeuron(neuron=temp_knowledge, activation=0.8),
        ActivatedNeuron(neuron=temp_tool, activation=0.7),
        ActivatedNeuron(neuron=calculator_tool, activation=0.6)
    ]
    
    # Execute all activated tools
    tool_results = query_engine.execute_tool_neurons(
        activated_neurons,
        execution_threshold=0.5
    )
    
    print(f"   - Total tools executed: {tool_results['total_executed']}")
    print(f"   - Total tools failed: {tool_results['total_failed']}")
    
    for i, exec_result in enumerate(tool_results['executed_tools'], 1):
        print(f"\n   Tool {i}:")
        print(f"     - Neuron ID: {exec_result['neuron_id']}")
        print(f"     - Activation: {exec_result['activation']:.3f}")
        print(f"     - Inputs: {exec_result['inputs']}")
        print(f"     - Result: {exec_result['result']}")
        print(f"     - Execution time: {exec_result['average_execution_time']:.2f}ms")
    
    # Aggregate results
    print("\n9. Aggregating Tool Results...")
    aggregated = query_engine.aggregate_tool_results(tool_results)
    print(f"   - Success rate: {aggregated['summary']['success_rate']:.1%}")
    print(f"   - Successful executions: {len(aggregated['successful_executions'])}")
    
    # Update a tool
    print("\n10. Updating Tool Neuron...")
    new_code = """
# Enhanced calculator with more operations
operation = inputs.get('operation', 'add')
a = inputs.get('a', 0)
b = inputs.get('b', 0)

operations = {
    'add': lambda x, y: x + y,
    'subtract': lambda x, y: x - y,
    'multiply': lambda x, y: x * y,
    'divide': lambda x, y: x / y if y != 0 else 'Error: Division by zero',
    'power': lambda x, y: x ** y,
    'modulo': lambda x, y: x % y if y != 0 else 'Error: Division by zero'
}

result = operations.get(operation, lambda x, y: 'Error: Unknown operation')(a, b)
"""
    
    success = training_engine.update_tool_neuron(
        calculator_id,
        new_code=new_code,
        new_description="Enhanced calculator with power and modulo operations"
    )
    
    if success:
        print("   - Tool updated successfully")
        
        # Test the updated tool
        updated_result = calculator_tool.execute({
            "operation": "power",
            "a": 2,
            "b": 8
        })
        print(f"   - Test new operation (2^8): {updated_result}")
    
    # Display statistics
    print("\n11. System Statistics...")
    print(f"   - Total neurons: {graph.get_neuron_count()}")
    print(f"   - Total synapses: {graph.get_synapse_count()}")
    print(f"   - Tool neurons: {sum(1 for n in graph.neurons.values() if n.neuron_type.value == 'tool')}")
    
    training_stats = training_engine.get_stats()
    print(f"   - Training operations: {training_stats['total_operations']}")
    
    query_stats = query_engine.get_performance_stats()
    print(f"   - Queries executed: {query_stats['total_queries']}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
