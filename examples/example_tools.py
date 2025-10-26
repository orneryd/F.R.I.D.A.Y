"""
Example Tool Neurons - Demonstrating tool integration.

This shows how to create and use Tool Neurons for:
- Web Search
- Calculator
- Weather API
- Custom Functions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron_system.neuron_types.tool_neuron import ToolNeuron
from neuron_system.core.vector3d import Vector3D
import numpy as np


def create_calculator_tool():
    """
    Create a calculator tool neuron.
    
    This tool can perform basic arithmetic operations.
    """
    calculator_code = """
# Calculator Tool
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
    if b == 0:
        result = {'error': 'Division by zero'}
    else:
        result = a / b
else:
    result = {'error': f'Unknown operation: {operation}'}
"""
    
    tool = ToolNeuron(
        function_signature="calculator(operation: str, a: float, b: float) -> float",
        executable_code=calculator_code,
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
        },
        output_schema={
            "type": "number"
        },
        activation_threshold=0.5
    )
    
    # Set position and vector
    tool.position = Vector3D(10.0, 10.0, 10.0)
    tool.vector = np.random.randn(384)
    
    return tool


def create_web_search_tool():
    """
    Create a web search tool neuron (placeholder).
    
    In production, this would integrate with a real search API.
    """
    search_code = """
# Web Search Tool (Placeholder)
query = inputs.get('query', '')
max_results = inputs.get('max_results', 5)

# In production, this would call a real search API
# For now, return a placeholder
result = {
    'query': query,
    'results': [
        {'title': f'Result {i+1}', 'url': f'https://example.com/{i+1}'}
        for i in range(max_results)
    ],
    'count': max_results
}
"""
    
    tool = ToolNeuron(
        function_signature="web_search(query: str, max_results: int = 5) -> dict",
        executable_code=search_code,
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"}
            },
            "required": ["query"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "results": {"type": "array"},
                "count": {"type": "integer"}
            }
        },
        activation_threshold=0.6
    )
    
    tool.position = Vector3D(20.0, 20.0, 20.0)
    tool.vector = np.random.randn(384)
    
    return tool


def create_weather_tool():
    """
    Create a weather API tool neuron (placeholder).
    
    In production, this would integrate with a real weather API.
    """
    weather_code = """
# Weather Tool (Placeholder)
location = inputs.get('location', 'Unknown')

# In production, this would call a real weather API
# For now, return a placeholder
result = {
    'location': location,
    'temperature': 20.0,
    'condition': 'Sunny',
    'humidity': 60,
    'wind_speed': 10.0
}
"""
    
    tool = ToolNeuron(
        function_signature="get_weather(location: str) -> dict",
        executable_code=weather_code,
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "temperature": {"type": "number"},
                "condition": {"type": "string"},
                "humidity": {"type": "integer"},
                "wind_speed": {"type": "number"}
            }
        },
        activation_threshold=0.5
    )
    
    tool.position = Vector3D(30.0, 30.0, 30.0)
    tool.vector = np.random.randn(384)
    
    return tool


def test_calculator_tool():
    """Test the calculator tool."""
    print("=" * 70)
    print("TESTING CALCULATOR TOOL")
    print("=" * 70)
    print()
    
    calc = create_calculator_tool()
    
    # Test addition
    result = calc.execute({'operation': 'add', 'a': 5, 'b': 3})
    print(f"5 + 3 = {result}")
    
    # Test multiplication
    result = calc.execute({'operation': 'multiply', 'a': 4, 'b': 7})
    print(f"4 * 7 = {result}")
    
    # Test division
    result = calc.execute({'operation': 'divide', 'a': 10, 'b': 2})
    print(f"10 / 2 = {result}")
    
    # Test division by zero
    result = calc.execute({'operation': 'divide', 'a': 10, 'b': 0})
    print(f"10 / 0 = {result}")
    
    print()
    print(f"Execution count: {calc.execution_count}")
    print(f"Average execution time: {calc.average_execution_time:.2f}ms")
    print()


def test_web_search_tool():
    """Test the web search tool."""
    print("=" * 70)
    print("TESTING WEB SEARCH TOOL")
    print("=" * 70)
    print()
    
    search = create_web_search_tool()
    
    result = search.execute({'query': 'artificial intelligence', 'max_results': 3})
    print(f"Query: {result['query']}")
    print(f"Results: {result['count']}")
    for i, res in enumerate(result['results'], 1):
        print(f"  {i}. {res['title']} - {res['url']}")
    
    print()
    print(f"Execution count: {search.execution_count}")
    print()


def test_weather_tool():
    """Test the weather tool."""
    print("=" * 70)
    print("TESTING WEATHER TOOL")
    print("=" * 70)
    print()
    
    weather = create_weather_tool()
    
    result = weather.execute({'location': 'Berlin'})
    print(f"Weather in {result['location']}:")
    print(f"  Temperature: {result['temperature']}°C")
    print(f"  Condition: {result['condition']}")
    print(f"  Humidity: {result['humidity']}%")
    print(f"  Wind Speed: {result['wind_speed']} km/h")
    
    print()
    print(f"Execution count: {weather.execution_count}")
    print()


if __name__ == "__main__":
    test_calculator_tool()
    test_web_search_tool()
    test_weather_tool()
    
    print("=" * 70)
    print("ALL TOOL TESTS COMPLETE")
    print("=" * 70)
    print()
    print("These tools can be integrated into the neuron network by:")
    print("1. Adding them to the graph")
    print("2. Creating connections to knowledge neurons")
    print("3. Activating them based on query context")
    print()
