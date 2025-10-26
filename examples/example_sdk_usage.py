"""
Example usage of the Neuron System Python SDK

This script demonstrates various SDK features including:
- Creating knowledge neurons
- Creating tool neurons
- Searching for knowledge
- Training neurons
- Managing connections
- Error handling
"""

from neuron_system.sdk import (
    NeuronSystemClient,
    NotFoundError,
    ValidationError,
    ConnectionError
)


def example_basic_usage():
    """Example 1: Basic usage of the SDK"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Initialize client
    client = NeuronSystemClient(base_url="http://localhost:8000")
    
    # Check system health
    health = client.health_check()
    print(f"\nSystem Status: {health['status']}")
    print(f"Neurons: {health['neuron_count']}")
    print(f"Synapses: {health['synapse_count']}")
    
    # Add knowledge
    neuron = client.add_knowledge(
        text="Python is a high-level, interpreted programming language",
        tags=["programming", "python", "language"]
    )
    print(f"\nCreated knowledge neuron: {neuron['id']}")
    print(f"Position: ({neuron['position']['x']:.2f}, "
          f"{neuron['position']['y']:.2f}, "
          f"{neuron['position']['z']:.2f})")
    
    # Search for knowledge
    results = client.search("programming language", limit=3)
    print(f"\nSearch results for 'programming language':")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Score: {result['score']:.3f}")
        print(f"     Content: {result['content'][:60]}...")
        print(f"     Tags: {result['tags']}")


def example_knowledge_base():
    """Example 2: Building a knowledge base"""
    print("\n" + "=" * 60)
    print("Example 2: Building a Knowledge Base")
    print("=" * 60)
    
    with NeuronSystemClient(base_url="http://localhost:8000") as client:
        # Add multiple related pieces of knowledge
        knowledge_items = [
            ("Machine learning is a subset of artificial intelligence", 
             ["AI", "ML", "technology"]),
            ("Neural networks are inspired by biological neurons", 
             ["AI", "neural-networks", "biology"]),
            ("Deep learning uses multiple layers of neural networks", 
             ["AI", "deep-learning", "neural-networks"]),
            ("Supervised learning requires labeled training data", 
             ["ML", "supervised-learning", "training"])
        ]
        
        neuron_ids = []
        print("\nAdding knowledge to the system:")
        for text, tags in knowledge_items:
            neuron = client.add_knowledge(text, tags=tags)
            neuron_ids.append(neuron['id'])
            print(f"  ✓ Added: {text[:50]}...")
        
        # Connect related concepts
        print("\nConnecting related concepts:")
        for i in range(len(neuron_ids) - 1):
            synapse = client.connect(
                from_neuron_id=neuron_ids[i],
                to_neuron_id=neuron_ids[i + 1],
                strength=0.7
            )
            print(f"  ✓ Connected neuron {i} -> {i+1} (weight: {synapse['weight']})")
        
        # Search the knowledge base
        print("\nSearching for 'neural networks':")
        results = client.search("neural networks", limit=3, depth=2)
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Score: {result['score']:.3f}")
            print(f"    Content: {result['content']}")
            print(f"    Tags: {', '.join(result['tags'])}")


def example_tool_creation():
    """Example 3: Creating and using tool neurons"""
    print("\n" + "=" * 60)
    print("Example 3: Creating Tool Neurons")
    print("=" * 60)
    
    client = NeuronSystemClient(base_url="http://localhost:8000")
    
    # Create a simple calculator tool
    print("\nCreating a multiplication tool:")
    result = client.add_tool(
        name="multiply(x, y)",
        description="Multiply two numbers together",
        code="result = x * y",
        input_schema={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "First number"},
                "y": {"type": "number", "description": "Second number"}
            },
            "required": ["x", "y"]
        },
        output_schema={
            "type": "number",
            "description": "Product of x and y"
        }
    )
    
    tool_id = result['details']['tool_neuron_id']
    print(f"  ✓ Created tool neuron: {tool_id}")
    print(f"  ✓ Auto-connected: {result['details']['auto_connected']}")
    
    # Add related knowledge
    print("\nAdding related knowledge:")
    knowledge = client.add_knowledge(
        text="Multiplication is a mathematical operation that combines numbers",
        tags=["math", "arithmetic", "multiplication"]
    )
    print(f"  ✓ Created knowledge neuron: {knowledge['id']}")
    
    # Connect knowledge to tool
    synapse = client.connect(
        from_neuron_id=knowledge['id'],
        to_neuron_id=tool_id,
        strength=0.9
    )
    print(f"  ✓ Connected knowledge to tool (weight: {synapse['weight']})")
    
    # Query to potentially activate the tool
    print("\nQuerying for 'multiply numbers':")
    results = client.search("multiply numbers", limit=5, depth=3)
    for result in results:
        print(f"  - {result['type']}: {result['content'][:50]}... "
              f"(score: {result['score']:.3f})")
        if result.get('tool_result'):
            print(f"    Tool executed: {result['tool_result']}")


def example_training():
    """Example 4: Training and updating knowledge"""
    print("\n" + "=" * 60)
    print("Example 4: Training and Updating Knowledge")
    print("=" * 60)
    
    client = NeuronSystemClient(base_url="http://localhost:8000")
    
    # Add initial knowledge
    print("\nAdding initial knowledge:")
    neuron = client.add_knowledge(
        text="Python 3.11 was released in October 2022",
        tags=["python", "version", "release"]
    )
    print(f"  ✓ Created neuron: {neuron['id']}")
    print(f"  ✓ Initial content: {neuron['source_data']}")
    
    # Update the knowledge with training
    print("\nTraining neuron with updated information:")
    result = client.train(
        neuron_id=neuron['id'],
        new_knowledge="Python 3.12 was released in October 2023",
        learning_rate=0.3  # Moderate learning rate
    )
    print(f"  ✓ Training successful: {result['success']}")
    print(f"  ✓ Operation ID: {result['operation_id']}")
    
    # Verify the update
    updated_neuron = client.get_neuron(neuron['id'])
    print(f"  ✓ Neuron updated (modified: {updated_neuron['modified_at']})")


def example_network_exploration():
    """Example 5: Exploring network connections"""
    print("\n" + "=" * 60)
    print("Example 5: Exploring Network Connections")
    print("=" * 60)
    
    client = NeuronSystemClient(base_url="http://localhost:8000")
    
    # Get network statistics
    stats = client.get_network_stats()
    print(f"\nNetwork Statistics:")
    print(f"  Total Neurons: {stats['neuron_count']}")
    print(f"  Total Synapses: {stats['synapse_count']}")
    print(f"  Status: {stats['status']}")
    
    # Create a central neuron
    print("\nCreating a central concept neuron:")
    central = client.add_knowledge(
        text="Artificial Intelligence is the simulation of human intelligence",
        tags=["AI", "technology", "intelligence"]
    )
    print(f"  ✓ Created: {central['id']}")
    
    # Create related neurons and connect them
    print("\nCreating related concepts:")
    related_concepts = [
        "Machine learning enables computers to learn from data",
        "Natural language processing helps computers understand human language",
        "Computer vision allows machines to interpret visual information"
    ]
    
    for concept in related_concepts:
        neuron = client.add_knowledge(concept, tags=["AI", "subfield"])
        synapse = client.connect(
            from_neuron_id=central['id'],
            to_neuron_id=neuron['id'],
            strength=0.8
        )
        print(f"  ✓ Connected: {concept[:40]}...")
    
    # Explore connections
    print(f"\nExploring connections from central neuron:")
    neighbors = client.get_neighbors(central['id'])
    print(f"  Found {neighbors['count']} connected neurons:")
    
    for neighbor_data in neighbors['neighbors']:
        synapse = neighbor_data['synapse']
        neighbor = neighbor_data['neuron']
        print(f"\n  Connection:")
        print(f"    Target: {neighbor['id']}")
        print(f"    Weight: {synapse['weight']}")
        print(f"    Usage: {synapse['usage_count']} times")
        if neighbor.get('source_data'):
            print(f"    Content: {neighbor['source_data'][:50]}...")


def example_connection_management():
    """Example 6: Managing connection strengths"""
    print("\n" + "=" * 60)
    print("Example 6: Managing Connection Strengths")
    print("=" * 60)
    
    client = NeuronSystemClient(base_url="http://localhost:8000")
    
    # Create two neurons
    print("\nCreating two related neurons:")
    neuron1 = client.add_knowledge(
        text="Reinforcement learning learns through trial and error",
        tags=["ML", "reinforcement-learning"]
    )
    neuron2 = client.add_knowledge(
        text="Q-learning is a reinforcement learning algorithm",
        tags=["ML", "reinforcement-learning", "algorithm"]
    )
    print(f"  ✓ Created neuron 1: {neuron1['id']}")
    print(f"  ✓ Created neuron 2: {neuron2['id']}")
    
    # Create connection
    synapse = client.connect(
        from_neuron_id=neuron1['id'],
        to_neuron_id=neuron2['id'],
        strength=0.5
    )
    print(f"\n  ✓ Created connection: {synapse['id']}")
    print(f"    Initial weight: {synapse['weight']}")
    
    # Strengthen the connection
    print("\nStrengthening the connection:")
    result = client.strengthen_connection(synapse['id'], amount=0.2)
    print(f"  ✓ Strengthened (new weight: {result['details']['final_weight']})")
    
    # Weaken the connection
    print("\nWeakening the connection:")
    result = client.weaken_connection(synapse['id'], amount=0.1)
    print(f"  ✓ Weakened (new weight: {result['details']['final_weight']})")


def example_error_handling():
    """Example 7: Error handling"""
    print("\n" + "=" * 60)
    print("Example 7: Error Handling")
    print("=" * 60)
    
    client = NeuronSystemClient(base_url="http://localhost:8000")
    
    # Example 1: Handle not found error
    print("\nTrying to get non-existent neuron:")
    try:
        neuron = client.get_neuron("00000000-0000-0000-0000-000000000000")
    except NotFoundError as e:
        print(f"  ✓ Caught NotFoundError: {e}")
    
    # Example 2: Handle validation error
    print("\nTrying to create neuron with invalid data:")
    try:
        neuron = client.create_neuron(
            neuron_type="invalid_type",
            source_data="Some data"
        )
    except ValidationError as e:
        print(f"  ✓ Caught ValidationError: {e}")
    
    # Example 3: Handle connection error (with wrong URL)
    print("\nTrying to connect to invalid URL:")
    try:
        bad_client = NeuronSystemClient(
            base_url="http://invalid-url:9999",
            timeout=2
        )
        bad_client.health_check()
    except ConnectionError as e:
        print(f"  ✓ Caught ConnectionError: {str(e)[:60]}...")
    
    print("\n  All error handling examples completed successfully!")


def example_batch_operations():
    """Example 8: Batch operations for performance"""
    print("\n" + "=" * 60)
    print("Example 8: Batch Operations")
    print("=" * 60)
    
    client = NeuronSystemClient(base_url="http://localhost:8000")
    
    # Create multiple neurons at once
    print("\nCreating 10 neurons in batch:")
    neurons_data = [
        {
            "neuron_type": "knowledge",
            "source_data": f"Knowledge item number {i}",
            "semantic_tags": ["batch", "example", f"item-{i}"]
        }
        for i in range(10)
    ]
    
    result = client.create_neurons_batch(neurons_data)
    print(f"  ✓ Created {result['count']} neurons")
    print(f"  ✓ IDs: {', '.join(result['created_ids'][:3])}...")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Neuron System SDK Examples")
    print("=" * 60)
    print("\nMake sure the API is running at http://localhost:8000")
    print("You can start it with: python run_api.py")
    
    try:
        # Run all examples
        example_basic_usage()
        example_knowledge_base()
        example_tool_creation()
        example_training()
        example_network_exploration()
        example_connection_management()
        example_error_handling()
        example_batch_operations()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except ConnectionError:
        print("\n" + "=" * 60)
        print("ERROR: Could not connect to the API")
        print("=" * 60)
        print("\nPlease make sure the API is running:")
        print("  python run_api.py")
        print("\nThen run this script again.")
    except Exception as e:
        print(f"\n" + "=" * 60)
        print(f"ERROR: {type(e).__name__}")
        print("=" * 60)
        print(f"\n{str(e)}")


if __name__ == "__main__":
    main()
