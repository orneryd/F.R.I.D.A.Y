# F.R.I.D.A.Y AI - Features & Capabilities

Vollständige Feature-Dokumentation für das 3D Synaptic Neuron System.

## Inhaltsverzeichnis

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
- **Räumliches Neuronen-System** mit 3D-Positionierung
- **Synaptische Verbindungen** mit Gewichten
- **Aktivierungs-Propagation** durch das Netzwerk
- **Octree-basierte Spatial Indexing** für schnelle Suche

### Neuron Types
```python
# Knowledge Neuron - Speichert Wissen
knowledge_neuron = KnowledgeNeuron(
    source_data="AI is artificial intelligence",
    semantic_tags=['ai', 'definition']
)

# Memory Neuron - Speichert Konversationen
memory_neuron = MemoryNeuron(
    conversation_id="conv_123",
    turn_number=1
)

# Tool Neuron - Führt Code aus
tool_neuron = ToolNeuron(
    tool_name="calculator",
    code="def add(a, b): return a + b"
)
```

### Vector Embeddings
- **Dynamische Dimensionen** (384D, 768D, 1024D)
- **Sentence-Transformers** für Text-Encoding
- **Cosine Similarity** für Relevanz-Suche
- **Normalisierung** für konsistente Vergleiche

---

## Neural Inference Engine

### Transformer-basierte Inferenz
Echte KI-Logik statt nur Vektor-Suche:

```python
from neuron_system.ai import SmartLanguageModel

model = SmartLanguageModel(
    graph, compression_engine, query_engine, training_engine,
    pretrained_model="distilbert-base-uncased"
)

# Nutzt Multi-Head Attention + Feed-Forward Networks
response = model.generate_response(
    "What is AI?",
    use_neural_inference=True
)
```

### Multi-Head Attention
- **6-12 Attention Heads** (abhängig von Dimension)
- **Kontextuelles Verständnis** statt nur Wort-Ähnlichkeit
- **Multi-Aspekt Reasoning** (Semantik, Kontext, Logik, etc.)
- **Pretrained Weights** von Hugging Face Modellen

### Feed-Forward Networks
- **2-Layer MLP** mit GELU Activation
- **Nicht-lineare Transformationen** für komplexe Muster
- **Hidden Dimension** = 4x Embedding Dimension
- **Layer Normalization** für Stabilität

### Supported Models
| Model | Dimensionen | Qualität | Speed |
|-------|-------------|----------|-------|
| DistilBERT | 768 → 384 | ⭐⭐⭐ | ⚡⚡⚡ |
| BERT | 768 | ⭐⭐⭐⭐ | ⚡⚡ |
| RoBERTa | 768 | ⭐⭐⭐⭐⭐ | ⚡ |
| RoBERTa-Large | 1024 | ⭐⭐⭐⭐⭐ | ⚡ |

---

## Training System

### Smart Trainer
Intelligentes Training mit Quality Control:

```python
from neuron_system.ai.training import SmartTrainer

trainer = SmartTrainer(language_model)

# Train mit automatischer Quality-Prüfung
success, reason = trainer.train_conversation(
    question="What is machine learning?",
    answer="Machine learning is a subset of AI..."
)
```

**Features:**
- ✅ **Quality Filtering** - Filtert schlechte Daten
- ✅ **Duplicate Detection** - Verhindert Duplikate
- ✅ **Logic Validation** - Prüft logische Konsistenz
- ✅ **Batch Processing** - Effizientes Batch-Training

### Incremental Trainer
Updates ohne komplettes Neutraining:

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
- ✅ **Tag Updates** für existierende Neuronen
- ✅ **Connection Updates**
- ✅ **Batch Operations**

### Self Training
Kontinuierliche Selbstverbesserung:

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
- ✅ **Auto-Evaluation** - Bewertet Antworten automatisch
- ✅ **Neuron Reinforcement** - Stärkt gute Neuronen
- ✅ **Neuron Weakening** - Schwächt schlechte Neuronen
- ✅ **Performance Tracking**

---

## Dimension System

### Flexible Dimensionen
System erkennt Dimensionen automatisch:

```python
# 384D (Standard)
compression_engine = CompressionEngine("all-MiniLM-L6-v2")
# → 384 Dimensionen

# 768D (Empfohlen)
compression_engine = CompressionEngine("all-mpnet-base-v2")
# → 768 Dimensionen

# 1024D (Maximum)
compression_engine = CompressionEngine("sentence-transformers/all-roberta-large-v1")
# → 1024 Dimensionen
```

### Dimension Upgrade
```bash
# Upgrade zu höheren Dimensionen
python scripts/migrate_to_higher_dimensions.py

# Wähle Konfiguration:
# [1] MEDIUM - 768D (+28% Qualität)
# [2] LARGE - 768D + RoBERTa
# [3] XLARGE - 1024D (Maximum)
```

### Vorteile höherer Dimensionen
| Dimension | Informationsdichte | Qualität | Speed |
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
# Automatisch GPU nutzen wenn verfügbar
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
- ✅ **CPU Fallback** (automatisch)

---

## Advanced Features

### Spatial Indexing
```python
# Octree-basierte räumliche Suche
from neuron_system.spatial import SpatialIndex

spatial_index = SpatialIndex(graph)
nearby_neurons = spatial_index.find_nearby(
    position=Vector3D(10, 20, 30),
    radius=5.0
)
```

### Activation Propagation
```python
# Aktivierung durch Netzwerk propagieren
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
# Hebbian Learning - Synapsen stärken/schwächen
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
# Tool Neurons können Code ausführen
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
# Text zu Vektor komprimieren
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
# Semantische Suche
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
# SQLite-basierte Persistenz
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
# Export zu JSON
graph.export_to_json("export.json")

# Import von JSON
graph.import_from_json("export.json")
```

---

## Performance Tuning

### Batch Size
```python
# Größere Batches = schneller (mit GPU)
compression_engine.compress_batch(
    texts,
    batch_size=256  # Default: 32
)
```

### Propagation Depth
```python
# Weniger Depth = schneller
results = query_engine.query(
    query_text="...",
    propagation_depth=1  # Statt 3
)
```

### Context Size
```python
# Weniger Kontext = schneller
response = model.generate_response(
    query,
    context_size=5  # Statt 10
)
```

### Caching
```python
# Model wird gecached (lazy loading)
compression_engine._ensure_model_loaded()  # Einmalig
# Danach: Schnelle Wiederverwendung
```

---

## Best Practices

### 1. Training
- ✅ Nutze `SmartTrainer` für Quality Control
- ✅ Batch-Training für Effizienz
- ✅ Regelmäßig `consolidate_learning()` aufrufen
- ✅ Statistics tracken

### 2. Dimensionen
- ✅ 384D für Tests & Entwicklung
- ✅ 768D für Produktion (beste Balance)
- ✅ 1024D für höchste Qualität

### 3. GPU
- ✅ GPU für Batch-Operations nutzen
- ✅ `use_gpu=True` setzen
- ✅ Batch-Size erhöhen mit GPU

### 4. Queries
- ✅ `propagation_depth=0` für schnelle Suche
- ✅ `propagation_depth=2-3` für bessere Qualität
- ✅ `min_activation` für Filtering

### 5. Memory
- ✅ Regelmäßig alte Memories löschen
- ✅ Synapse decay für unused connections
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
# System erkennt Dimensionen automatisch
# Stelle sicher dass alle Modelle kompatibel sind
compression_engine = CompressionEngine("all-MiniLM-L6-v2")  # 384D
# Neural Engine passt sich automatisch an
```

### "Out of memory"
```python
# Reduziere Batch-Size
compression_engine.compress_batch(texts, batch_size=16)

# Oder nutze kleineres Modell
compression_engine = CompressionEngine("all-MiniLM-L6-v2")  # Statt all-mpnet-base-v2
```

### "Slow inference"
```python
# Nutze GPU
compression_engine = CompressionEngine(use_gpu=True)

# Reduziere Context
model.generate_response(query, context_size=5)

# Reduziere Propagation
query_engine.query(text, propagation_depth=1)
```

---

## Examples

Siehe `examples/` Ordner für vollständige Beispiele:
- `neural_inference_demo.py` - Neural Inference Demo
- Weitere Beispiele in Entwicklung

## Scripts

Siehe `scripts/` Ordner für Utility-Scripts:
- `setup_neural_inference.py` - Setup Neural Inference
- `migrate_to_higher_dimensions.py` - Dimension Upgrade
- `test_dynamic_dimensions.py` - Test Dimensionen
- `benchmark_gpu.py` - GPU Benchmarks

---

## Support

Bei Fragen oder Problemen:
1. Prüfe diese Dokumentation
2. Siehe `CLI.md` für CLI-Befehle
3. Siehe `CHANGELOG.md` für Updates
4. Check GitHub Issues

---

**Version:** 2.2 (Neural Inference)
**Last Updated:** Oktober 2025
