"""
Neural Inference Demo - Zeigt wie man echte KI-Logik nutzt.

Dieses Script demonstriert:
1. Laden eines vortrainierten Modells
2. Integration in das Neuron System
3. Intelligente Antworten mit Neural Inference
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from neuron_system.core.graph import NeuronGraph
from neuron_system.engines.compression import CompressionEngine
from neuron_system.engines.query import QueryEngine
from neuron_system.engines.training import TrainingEngine
from neuron_system.ai.smart_language_model import SmartLanguageModel
from neuron_system.ai.model_loader import ModelLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    
    print("\n" + "=" * 70)
    print("NEURAL INFERENCE ENGINE DEMO")
    print("=" * 70 + "\n")
    
    # 1. Show available models
    print("Step 1: Available Pretrained Models")
    print("-" * 70)
    ModelLoader.print_available_models()
    
    # 2. Initialize system
    print("\nStep 2: Initializing Neuron System")
    print("-" * 70)
    
    graph = NeuronGraph()
    compression_engine = CompressionEngine()
    query_engine = QueryEngine(graph, compression_engine)
    training_engine = TrainingEngine(graph)
    
    print("[OK] Neuron system initialized\n")
    
    # 3. Create Smart Language Model with Neural Inference
    print("Step 3: Creating Smart Language Model")
    print("-" * 70)
    
    try:
        # Use DistilBERT (fast and small)
        model = SmartLanguageModel(
            graph=graph,
            compression_engine=compression_engine,
            query_engine=query_engine,
            training_engine=training_engine,
            enable_self_training=True,
            pretrained_model="distilbert-base-uncased"
        )
        
        print("[OK] Smart Language Model created with Neural Inference\n")
        
    except Exception as e:
        logger.error(f"Failed to create Smart Language Model: {e}")
        logger.info("Falling back to standard Language Model")
        
        from neuron_system.ai.language_model import LanguageModel
        model = LanguageModel(
            graph, compression_engine, query_engine, 
            training_engine, enable_self_training=True
        )
        print("[OK] Standard Language Model created\n")
    
    # 4. Add some knowledge
    print("Step 4: Adding Knowledge")
    print("-" * 70)
    
    knowledge_items = [
        {
            'text': 'Question: What is artificial intelligence? Answer: Artificial intelligence (AI) is the simulation of human intelligence by machines, especially computer systems.',
            'tags': ['ai', 'definition', 'qa']
        },
        {
            'text': 'Question: What is a neural network? Answer: A neural network is a computing system inspired by biological neural networks that learns to perform tasks by considering examples.',
            'tags': ['neural-network', 'definition', 'qa']
        },
        {
            'text': 'Question: What is machine learning? Answer: Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.',
            'tags': ['machine-learning', 'definition', 'qa']
        },
        {
            'text': 'Question: What is deep learning? Answer: Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn hierarchical representations.',
            'tags': ['deep-learning', 'definition', 'qa']
        },
        {
            'text': 'Question: What is natural language processing? Answer: Natural language processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language.',
            'tags': ['nlp', 'definition', 'qa']
        },
    ]
    
    for item in knowledge_items:
        model.learn(text=item['text'], tags=item['tags'])
        print(f"  [OK] Learned: {item['text'][:60]}...")
    
    print(f"\n[OK] Added {len(knowledge_items)} knowledge items\n")
    
    # 5. Test queries
    print("Step 5: Testing Neural Inference")
    print("-" * 70 + "\n")
    
    test_queries = [
        "What is AI?",
        "Explain neural networks",
        "Tell me about machine learning",
        "What is NLP?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 40)
        
        # Generate response with neural inference
        response = model.generate_response(
            query,
            context_size=5,
            use_neural_inference=True,
            use_reasoning=True
        )
        
        print(f"Response: {response}")
        print()
    
    # 6. Show statistics
    print("\nStep 6: Statistics")
    print("-" * 70)
    
    if hasattr(model, 'get_neural_statistics'):
        stats = model.get_neural_statistics()
    else:
        stats = model.get_statistics()
    
    print(f"Total Neurons: {stats['total_neurons']}")
    print(f"Knowledge Neurons: {stats['knowledge_neurons']}")
    print(f"Total Synapses: {stats['total_synapses']}")
    print(f"Average Connectivity: {stats['average_connectivity']:.2f}")
    
    if 'neural_inference_enabled' in stats:
        print(f"\nNeural Inference: {'[ENABLED]' if stats['neural_inference_enabled'] else '[DISABLED]'}")
        
        if stats['neural_inference_enabled'] and 'neural_engine_info' in stats:
            info = stats['neural_engine_info']
            print(f"  Embedding Dim: {info['embedding_dim']}")
            print(f"  Attention Heads: {info['num_attention_heads']}")
            print(f"  Hidden Dim: {info['hidden_dim']}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
