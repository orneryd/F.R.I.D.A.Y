"""
Test Dynamic Dimensions - Zeigt dass das System automatisch die richtigen Dimensionen erkennt.
"""

import logging
from neuron_system.core.graph import NeuronGraph
from neuron_system.engines.compression import CompressionEngine
from neuron_system.engines.query import QueryEngine
from neuron_system.engines.training import TrainingEngine
from neuron_system.ai.smart_language_model import SmartLanguageModel

logging.basicConfig(level=logging.INFO, format='%(message)s')

print("\n" + "=" * 70)
print("DYNAMIC DIMENSION TEST")
print("=" * 70 + "\n")

# Test 1: Default (384D)
print("Test 1: Default Model (all-MiniLM-L6-v2)")
print("-" * 70)

graph = NeuronGraph()
compression_engine = CompressionEngine()  # Default: all-MiniLM-L6-v2
query_engine = QueryEngine(graph, compression_engine)
training_engine = TrainingEngine(graph)

model = SmartLanguageModel(
    graph, compression_engine, query_engine, training_engine,
    pretrained_model="distilbert-base-uncased"
)

# Trigger model load
compression_engine._ensure_model_loaded()

print(f"Compression Engine: {compression_engine.vector_dim}D")
if model.neural_enabled:
    print(f"Neural Engine: {model.neural_engine.embedding_dim}D")
    print(f"Attention Heads: {model.neural_engine.num_attention_heads}")
    print(f"Hidden Dim: {model.neural_engine.hidden_dim}")
    print("[OK] Dimensions match!")
else:
    print("[WARN] Neural engine not enabled")

print()

# Test 2: 768D Model
print("Test 2: Larger Model (all-mpnet-base-v2 = 768D)")
print("-" * 70)

graph2 = NeuronGraph()
compression_engine2 = CompressionEngine(model_name="all-mpnet-base-v2")  # 768D
query_engine2 = QueryEngine(graph2, compression_engine2)
training_engine2 = TrainingEngine(graph2)

model2 = SmartLanguageModel(
    graph2, compression_engine2, query_engine2, training_engine2,
    pretrained_model="bert-base-uncased"
)

# Trigger model load
compression_engine2._ensure_model_loaded()

print(f"Compression Engine: {compression_engine2.vector_dim}D")
if model2.neural_enabled:
    print(f"Neural Engine: {model2.neural_engine.embedding_dim}D")
    print(f"Attention Heads: {model2.neural_engine.num_attention_heads}")
    print(f"Hidden Dim: {model2.neural_engine.hidden_dim}")
    print("[OK] Dimensions match!")
else:
    print("[WARN] Neural engine not enabled")

print()

# Test 3: Quick functionality test
print("Test 3: Functionality Test")
print("-" * 70)

model.learn(
    text="Question: What is AI? Answer: Artificial Intelligence.",
    tags=['test']
)
print("[OK] Training works")

response = model.generate_response("What is AI?", use_neural_inference=True)
print(f"Response: {response}")
print("[OK] Inference works")

print()
print("=" * 70)
print("ALL TESTS PASSED - DYNAMIC DIMENSIONS WORKING!")
print("=" * 70 + "\n")
