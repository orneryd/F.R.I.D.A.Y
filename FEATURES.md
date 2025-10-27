# F.R.I.D.A.Y AI - Features & Capabilities

Complete feature documentation for the 3D Synaptic Neuron System.

## Table of Contents

- [Core Features](#core-features)
- [Neural Inference Engine](#neural-inference-engine)
- [Training System](#training-system)
- [Dimension System](#dimension-system)
- [API](#api)
- [GPU Acceleration](#gpu-acceleration)
- [Advanced Features](#advanced-features)

---

## Core Features

### 3D Neuron Architecture
- **Spatial Neuron System** with 3D positioning
- **Synaptic Connections** with weights
- **Activation Propagation** through the network
- **Octree-based Spatial Indexing** for fast search

### Neuron Types
```python
# Knowledge Neuron - Stores knowledge
knowledge_neuron = KnowledgeNeuron(
    source_data="AI is artificial intelligence",
    semantic_tags=['ai', 'definition']
)

# Memory Neuron - Stores conversations
memory_neuron = MemoryNeuron(
    conversation_id="conv_123",
    turn_number=1
)

# Tool Neuron - Executes code
tool_neuron = ToolNeuron(
    tool_name="calculator",
    code="def add(a, b): return a + b"
)
```

### Vector Embeddings
- **Dynamic Dimensions** (384D, 768D, 1024D)
- **Sentence-Transformers** for text encoding
- **Cosine Similarity** for relevance search
- **Normalization** for consistent comparisons

---

## Neural Inference Engine

### Transformer-based Inference
Real AI logic instead of just vector search:

```python
from neuron_system.ai import SmartLanguageModel

model = SmartLanguageModel(
    graph, compression_engine, query_engine, training_engine,
    pretrained_model="distilbert-base-uncased"
)

# Uses Multi-Head Attention + Feed-Forward Networks
response = model.generate_response(
    "What is AI?",
    use_neural_inference=True
)
```

### Multi-Head Attention
- **6-12 Attention Heads** (depending on dimension)
- **Contextual Understanding** instead of just word similarity
- **Multi-Aspect Reasoning** (Semantics, Context, Logic, etc.)
- **Pretrained Weights** from Hugging Face models

### Feed-Forward Networks
- **2-Layer MLP** with GELU Activation
- **Non-linear Transformations** for complex patterns
- **Hidden Dimension** = 4x Embedding Dimension
- **Layer Normalization** for stability

### Supported Models
| Model | Dimensions | Quality | Speed |
|-------|-------------|----------|-------|
| DistilBERT | 768 → 384 | ⭐⭐⭐ | ⚡⚡⚡ |
| BERT | 768 | ⭐⭐⭐⭐ | ⚡⚡ |
| RoBERTa | 768 | ⭐⭐⭐⭐⭐ | ⚡ |
| RoBERTa-Large | 1024 | ⭐⭐⭐⭐⭐ | ⚡ |

---

## Training System

### Smart Trainer
Intelligent training with Quality Control:

```python
from neuron_system.ai.training import SmartTrainer

trainer = SmartTrainer(language_model)

# Train with automatic quality checking
success, reason = trainer.train_conversation(
    question="What is machine learning?",
    answer="Machine learning is a subset of AI..."
)
```

**Features:**
- ✅ **Quality Filtering** - Filters bad data
- ✅ **Duplicate Detection** - Prevents duplicates
- ✅ **Logic Validation** - Checks logical consistency
- ✅ **Batch Processing** - Efficient batch training

### Incremental Trainer
Updates without complete retraining:

```python
from neuron_system.ai.training import IncrementalTrainer

trainer = IncrementalTrainer(language_model)

# Add or update knowledge
neuron_id, was_updated = trainer.add_or_update_knowledge(
    text="New information...",
    tags=['category']
)
```

**Features:**
- ✅ **Similarity-based Deduplication**
- ✅ **Tag Updates** for existing neurons
- ✅ **Connection Updates**
- ✅ **Batch Operations**

### Self Training
Continuous self-improvement:

```python
from neuron_system.ai.training import SelfTraining

self_trainer = SelfTraining(language_model)

# Learn from responses
self_trainer.learn_from_response(
    query="What is AI?",
    response="Artificial Intelligence is...",
    activated_neurons=results,
    feedback="positive"  # or "auto"
)
```

**Features:**
- ✅ **Auto-Evaluation** - Evaluates responses automatically
- ✅ **Neuron Reinforcement** - Strengthens good neurons
- ✅ **Neuron Weakening** - Weakens bad neurons
- ✅ **Performance Tracking**

---

## Dimension System

### Flexible Dimensions
System detects dimensions automatically:

```python
# 384D (Standard)
compression_engine = CompressionEngine("all-MiniLM-L6-v2")
# → 384 dimensions

# 768D (Recommended)
compression_engine = CompressionEngine("all-mpnet-base-v2")
# → 768 dimensions

# 1024D (Maximum)
compression_engine = CompressionEngine("sentence-transformers/all-roberta-large-v1")
# → 1024 dimensions
```

### Dimension Upgrade
```bash
# Upgrade to higher dimensions
python scripts/migrate_to_higher_dimensions.py

# Choose configuration:
# [1] MEDIUM - 768D (+28% quality)
# [2] LARGE - 768D + RoBERTa
# [3] XLARGE - 1024D (Maximum)
```

### Benefits of Higher Dimensions
| Dimension | Information Density | Quality | Speed |
|-----------|-------------------|----------|-------|
| 384D | 1x | ⭐⭐⭐ | ⚡⚡⚡ |
| 768D | 2x | ⭐⭐⭐⭐ | ⚡⚡ |
| 1024D | 2.7x | ⭐⭐⭐⭐⭐ | ⚡ |

---

## API

### REST API
```python
# Start API Server
python run_api.py
```

**Endpoints:**

#### Query
```bash
POST /query
{
  "query_text": "What is AI?",
  "top_k": 5,
  "propagation_depth": 2
}
```

#### Learn
```bash
POST /learn
{
  "text": "AI is artificial intelligence",
  "tags": ["ai", "definition"]
}
```

#### Train
```bash
POST /train/conversation
{
  "question": "What is ML?",
  "answer": "Machine learning is..."
}
```

#### Statistics
```bash
GET /stats
```

### Python SDK
```python
from neuron_system.sdk import NeuronSystemClient

client = NeuronSystemClient("http://localhost:8000")

# Query
results = client.query("What is AI?", top_k=5)

# Learn
neuron_id = client.learn("New knowledge", tags=['category'])

# Train
success = client.train_conversation(
    question="What is AI?",
    answer="Artificial Intelligence is..."
)
```

---

## GPU Acceleration

### Automatic GPU Detection
```python
# Automatically use GPU if available
compression_engine = CompressionEngine(use_gpu=True)

# GPU Info
python cli.py gpu-info
```

### Performance
| Operation | CPU | GPU (CUDA) | Speedup |
|-----------|-----|------------|---------|
| Compression | 10ms | 2ms | 5x |
| Batch (100) | 1000ms | 100ms | 10x |
| Training | 60s | 6s | 10x |

### Supported Platforms
- ✅ **NVIDIA CUDA** (GeForce, RTX, Tesla)
- ✅ **Apple Silicon MPS** (M1, M2, M3)
- ✅ **CPU Fallback** (automatic)

---

## Advanced Features

### Spatial Indexing
```python
# Octree-based spatial search
from neuron_system.spatial import SpatialIndex

spatial_index = SpatialIndex(graph)
nearby_neurons = spatial_index.query_radius(
    position=Vector3D(10, 20, 30),
    radius=5.0
)
```

### Activation Propagation
```python
# Propagate activation through network
from neuron_system.engines.activation import ActivationEngine

activation_engine = ActivationEngine(graph)
activated = activation_engine.propagate(
    initial_neurons=[neuron1, neuron2],
    depth=3,
    threshold=0.5
)
```

### Synapse Learning
```python
# Hebbian Learning - strengthen/weaken synapses
from neuron_system.engines.training import TrainingEngine

training_engine = TrainingEngine(graph)

# Strengthen synapse (Hebbian)
training_engine.strengthen_synapse(synapse_id, delta=0.01)

# Weaken synapse (decay)
training_engine.weaken_synapse(synapse_id, delta=0.01)

# Usage-based learning
training_engine.apply_usage_based_learning(synapse_id)
```

### Memory System
```python
# Conversation Memory
from neuron_system.neuron_types import MemoryNeuron

memory = MemoryNeuron(
    conversation_id="conv_123",
    turn_number=1,
    speaker="user",
    content="Hello, how are you?"
)

graph.add_neuron(memory)
```

### Tool Execution
```python
# Tool Neurons can execute code
from neuron_system.neuron_types import ToolNeuron

tool = ToolNeuron(
    tool_name="calculator",
    code="""
def calculate(expression):
    return eval(expression)
""",
    input_schema={"expression": "string"},
    output_schema={"result": "number"}
)

# Execute
result = tool.execute({"expression": "2 + 2"})
# → {"result": 4}
```

### Compression Engine
```python
# Compress text to vector
from neuron_system.engines.compression import CompressionEngine

engine = CompressionEngine()
vector, metadata = engine.compress("Hello world")

# Batch compression
vectors, metadata_list = engine.compress_batch([
    "Text 1",
    "Text 2",
    "Text 3"
])
```

### Query Engine
```python
# Semantic search
from neuron_system.engines.query import QueryEngine

query_engine = QueryEngine(graph, compression_engine)
results = query_engine.query(
    query_text="What is AI?",
    top_k=10,
    propagation_depth=2,
    min_activation=0.3
)

for result in results:
    print(f"Neuron: {result.neuron.id}")
    print(f"Activation: {result.activation}")
    print(f"Distance: {result.distance}")
```

### Storage & Persistence
```python
# SQLite-based persistence
from neuron_system.storage import DatabaseManager

db = DatabaseManager("neuron_system.db")
graph.attach_storage(db)

# Auto-save
graph.save()

# Load
graph.load()
```

### Export & Import
```python
# Export to JSON
graph.export_to_json("export.json")

# Import from JSON
graph.import_from_json("export.json")
```

---

## Performance Tuning

### Batch Size
```python
# Larger batches = faster (with GPU)
compression_engine.compress_batch(
    texts,
    batch_size=256  # Default: 32
)
```

### Propagation Depth
```python
# Less depth = faster
results = query_engine.query(
    query_text="...",
    propagation_depth=1  # Instead of 3
)
```

### Context Size
```python
# Less context = faster
response = model.generate_response(
    query,
    context_size=5  # Instead of 10
)
```

### Caching
```python
# Model is cached (lazy loading)
compression_engine._ensure_model_loaded()  # One-time
# After that: Fast reuse
```

---

## Best Practices

### 1. Training
- ✅ Use `SmartTrainer` for Quality Control
- ✅ Batch training for efficiency
- ✅ Regularly call `consolidate_learning()`
- ✅ Track statistics

### 2. Dimensions
- ✅ 384D for testing & development
- ✅ 768D for production (best balance)
- ✅ 1024D for highest quality

### 3. GPU
- ✅ Use GPU for batch operations
- ✅ Set `use_gpu=True`
- ✅ Increase batch size with GPU

### 4. Queries
- ✅ `propagation_depth=0` for fast search
- ✅ `propagation_depth=2-3` for better quality
- ✅ `min_activation` for filtering

### 5. Memory
- ✅ Regularly delete old memories
- ✅ Synapse decay for unused connections
- ✅ Database cleanup

---

## Troubleshooting

### "No GPU detected"
```bash
# Check GPU
python cli.py gpu-info

# Install CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "Vector dimension mismatch"
```python
# System detects dimensions automatically
# Make sure all models are compatible
compression_engine = CompressionEngine("all-MiniLM-L6-v2")  # 384D
# Neural Engine adapts automatically
```

### "Out of memory"
```python
# Reduce batch size
compression_engine.compress_batch(texts, batch_size=16)

# Or use smaller model
compression_engine = CompressionEngine("all-MiniLM-L6-v2")  # Instead of all-mpnet-base-v2
```

### "Slow inference"
```python
# Use GPU
compression_engine = CompressionEngine(use_gpu=True)

# Reduce context
model.generate_response(query, context_size=5)

# Reduce propagation
query_engine.query(text, propagation_depth=1)
```

---

## Examples

See `examples/` folder for complete examples:
- `neural_inference_demo.py` - Neural Inference Demo
- More examples in development

## Scripts

See `scripts/` folder for utility scripts:
- `setup_neural_inference.py` - Setup Neural Inference
- `migrate_to_higher_dimensions.py` - Dimension Upgrade
- `test_dynamic_dimensions.py` - Test Dimensions
- `benchmark_gpu.py` - GPU Benchmarks

---

## Support

For questions or problems:
1. Check this documentation
2. See `CLI.md` for CLI commands
3. See `CHANGELOG.md` for updates
4. Check GitHub Issues

---

**Version:** 2.2 (Neural Inference)
**Last Updated:** October 2025
