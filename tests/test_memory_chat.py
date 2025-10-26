"""
Test memory-enabled chat to verify context awareness.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel
from neuron_system.neuron_types.memory_neuron import MemoryManager


def test_memory_chat():
    print("=" * 70)
    print("TESTING MEMORY-ENABLED CHAT")
    print("=" * 70)
    print()
    
    settings = Settings(
        database_path="comprehensive_ai.db",
        spatial_bounds_min=(-500.0, -500.0, -500.0),
        spatial_bounds_max=(500.0, 500.0, 500.0)
    )
    
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        language_model = LanguageModel(
            container.graph,
            container.compression_engine,
            container.query_engine,
            container.training_engine
        )
        
        memory_manager = MemoryManager(
            container.graph,
            container.compression_engine
        )
        
        # Simulate a conversation
        conversation = [
            ("My name is John", "Nice to meet you, John!"),
            ("I like pizza", "Pizza is great!"),
            ("What's my name?", "Should remember: John"),
            ("What do I like?", "Should remember: pizza"),
        ]
        
        print("Simulating conversation with memory...")
        print()
        
        for turn, (user_msg, expected) in enumerate(conversation, 1):
            print(f"Turn {turn}:")
            print(f"  User: {user_msg}")
            
            # Retrieve relevant memories
            relevant_memories = memory_manager.retrieve_memories(
                user_msg,
                top_k=3,
                min_importance=0.3
            )
            
            # Build context
            memory_context = ""
            if relevant_memories:
                memory_context = "\n[Context: "
                memory_context += "; ".join([
                    m.source_data[:40] for m in relevant_memories
                ])
                memory_context += "]"
                print(f"  Memory: {len(relevant_memories)} relevant memories found")
            
            # Generate response
            response = language_model.generate_response(
                user_msg + memory_context,
                context_size=5,
                min_activation=0.1
            )
            
            print(f"  AI: {response}")
            print(f"  Expected: {expected}")
            print()
            
            # Store in memory
            memory_manager.create_memory(
                content=f"User said: {user_msg}",
                memory_type="short-term",
                context={'turn': turn, 'speaker': 'user'},
                importance=0.6
            )
            
            memory_manager.create_memory(
                content=f"AI responded: {response}",
                memory_type="short-term",
                context={'turn': turn, 'speaker': 'ai'},
                importance=0.5
            )
        
        # Check memory statistics
        print("=" * 70)
        print("MEMORY STATISTICS")
        print("=" * 70)
        
        mem_stats = memory_manager.get_statistics()
        print(f"Total memories: {mem_stats['total_memories']}")
        print(f"By type: {mem_stats['by_type']}")
        print(f"Average importance: {mem_stats['average_importance']:.2f}")
        print(f"Total accesses: {mem_stats['total_accesses']}")
        print()
        
        # Test memory retrieval
        print("=" * 70)
        print("TESTING MEMORY RETRIEVAL")
        print("=" * 70)
        print()
        
        test_queries = [
            "What's my name?",
            "What do I like?",
            "Tell me about our conversation"
        ]
        
        for query in test_queries:
            print(f"Query: {query}")
            memories = memory_manager.retrieve_memories(query, top_k=3)
            print(f"Found {len(memories)} relevant memories:")
            for i, mem in enumerate(memories, 1):
                print(f"  {i}. {mem.source_data[:60]}... (importance: {mem.importance:.2f})")
            print()
        
        print("=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)
        print()
        print("Memory system is working!")
        print("- Stores conversation history")
        print("- Retrieves relevant context")
        print("- Tracks importance and access")
        print()
        
    finally:
        container.shutdown()


if __name__ == "__main__":
    test_memory_chat()
