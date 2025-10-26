

# AI Language Model Documentation

## Überblick

Das AI Language Model ist eine KI, die auf dem 3D Synaptic Neuron System aufbaut. Sie verwendet vortrainierte Knowledge Neurons, um natürliche Sprache zu verstehen und zu verarbeiten.

## Features

### 1. Pre-trained English Knowledge Base

Die KI kommt mit vortrainiertem Englisch-Wissen:

- **Vocabulary** (80+ Wörter): Grundlegende englische Wörter und Bedeutungen
- **Grammar**: Grammatikregeln und Sprachstruktur
- **Common Phrases**: Häufige Ausdrücke und Redewendungen
- **Facts**: Allgemeinwissen über Sprache
- **Numbers**: Zahlen und Zählen
- **Time & Dates**: Zeitbegriffe
- **Colors**: Farben und Beschreibungen
- **Actions**: Verben und Aktionen

### 2. Natural Language Understanding

Die KI kann:
- Fragen verstehen
- Relevante Neuronen aktivieren
- Kontext aus mehreren Neuronen kombinieren
- Antworten basierend auf aktiviertem Wissen generieren

### 3. Learning Capabilities

Die KI kann:
- Neues Wissen lernen
- Verbindungen zwischen verwandten Konzepten erstellen
- Durch Feedback verstärkt lernen
- Wissen persistent speichern

## Quick Start

### 1. Einfaches Beispiel

```python
from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel
from neuron_system.ai.pretraining import PreTrainingLoader

# System initialisieren
settings = Settings(database_path="my_ai.db")
container = ApplicationContainer(settings)
container.initialize()

# Language Model erstellen
language_model = LanguageModel(
    graph=container.graph,
    compression_engine=container.compression_engine,
    query_engine=container.query_engine,
    training_engine=container.training_engine
)

# Englisch-Wissen laden
loader = PreTrainingLoader(language_model)
loader.load_english_knowledge()

# Mit der KI interagieren
response = language_model.generate_response("What is hello?")
print(response)

# Cleanup
container.shutdown()
```

### 2. Interaktive Chat-Session

```bash
# Starte das vortrainierte AI Beispiel
python examples/example_pretrained_ai.py

# Oder im Demo-Modus
python examples/example_pretrained_ai.py --demo
```

## Verwendung

### Language Model Initialisierung

```python
from neuron_system.ai.language_model import LanguageModel

language_model = LanguageModel(
    graph=container.graph,
    compression_engine=container.compression_engine,
    query_engine=container.query_engine,
    training_engine=container.training_engine
)
```

### Pre-Training Laden

```python
from neuron_system.ai.pretraining import PreTrainingLoader

# Loader erstellen
loader = PreTrainingLoader(
    language_model=language_model,
    spatial_distribution="clustered"  # oder "grid", "random"
)

# Englisch-Wissen laden
neurons_created = loader.load_english_knowledge(
    create_connections=True,  # Synapsen zwischen verwandten Neuronen erstellen
    batch_size=10  # Fortschritt alle 10 Neuronen loggen
)

print(f"Loaded {neurons_created} neurons")
```

### Fragen Stellen

```python
# Einfache Frage
response = language_model.generate_response("What is hello?")
print(response)

# Mit mehr Kontext
response = language_model.generate_response(
    query="Tell me about colors",
    context_size=5,  # Verwende bis zu 5 Neuronen für Kontext
    min_activation=0.3  # Mindest-Aktivierung für Relevanz
)
print(response)
```

### Verstehen (Understanding)

```python
# Finde relevante Neuronen für einen Text
results = language_model.understand(
    text="What is a sentence?",
    top_k=10,  # Top 10 Ergebnisse
    propagation_depth=3  # Aktivierungs-Tiefe
)

# Ergebnisse anzeigen
for result in results:
    print(f"Neuron: {result.neuron.id}")
    print(f"Activation: {result.activation:.2f}")
    if hasattr(result.neuron, 'source_data'):
        print(f"Content: {result.neuron.source_data}")
    print()
```

### Neues Wissen Lernen

```python
# Neues Wissen hinzufügen
neuron_id = language_model.learn(
    text="Python is a programming language used for AI and web development",
    tags=["programming", "python", "technology"],
    create_connections=True  # Verbindungen zu verwandten Neuronen erstellen
)

print(f"Created neuron: {neuron_id}")
```

### Reinforcement Learning

```python
# Lernen durch Feedback verstärken
language_model.reinforce_learning(
    query="What is Python?",
    correct_response="Python is a programming language",
    learning_rate=0.1
)
```

### Statistiken

```python
stats = language_model.get_statistics()
print(f"Total Neurons: {stats['total_neurons']}")
print(f"Knowledge Neurons: {stats['knowledge_neurons']}")
print(f"Total Synapses: {stats['total_synapses']}")
print(f"Average Connectivity: {stats['average_connectivity']:.2f}")
```

## Spatial Distribution Strategien

### Grid Distribution
Verteilt Neuronen gleichmäßig in einem 3D-Gitter:
```python
loader = PreTrainingLoader(language_model, spatial_distribution="grid")
```

### Random Distribution
Verteilt Neuronen zufällig im Raum:
```python
loader = PreTrainingLoader(language_model, spatial_distribution="random")
```

### Clustered Distribution (Empfohlen)
Gruppiert verwandte Neuronen in Clustern:
```python
loader = PreTrainingLoader(language_model, spatial_distribution="clustered")
```

## Eigene Knowledge Base Laden

### Aus JSON-Datei

```python
# JSON Format:
# [
#   {"text": "...", "tags": ["tag1", "tag2"]},
#   {"text": "...", "tags": ["tag3"]}
# ]

loader.load_from_file(
    file_path="my_knowledge.json",
    create_connections=True
)
```

### Knowledge Base Speichern

```python
# Aktuelle Knowledge Base speichern
loader.save_knowledge_base("saved_knowledge.json")
```

## Erweiterte Verwendung

### Custom Response Generation

```python
class CustomLanguageModel(LanguageModel):
    def _synthesize_response(self, query, knowledge_pieces):
        # Custom response logic
        if not knowledge_pieces:
            return "I don't know."
        
        # Combine knowledge in a custom way
        texts = [p['text'] for p in knowledge_pieces]
        return " ".join(texts)

# Verwenden
custom_model = CustomLanguageModel(
    graph=container.graph,
    compression_engine=container.compression_engine,
    query_engine=container.query_engine,
    training_engine=container.training_engine
)
```

### Integration mit API

```python
from fastapi import FastAPI
from neuron_system.ai.language_model import LanguageModel

app = FastAPI()

# Language model als globale Variable
language_model = None

@app.on_event("startup")
async def startup():
    global language_model
    # Initialize language model
    # ...

@app.post("/ask")
async def ask_question(question: str):
    response = language_model.generate_response(question)
    return {"response": response}
```

## Performance Tipps

### 1. Batch Loading
Lade Wissen in Batches für bessere Performance:
```python
loader.load_english_knowledge(batch_size=50)
```

### 2. Connection Creation
Erstelle Verbindungen nach dem Laden aller Neuronen:
```python
loader.load_english_knowledge(create_connections=False)
# ... lade mehr Wissen ...
loader._create_all_connections()
```

### 3. Database Optimization
Verwende eine dedizierte Datenbank für die KI:
```python
settings = Settings(
    database_path="ai_knowledge.db",
    enable_auto_save=True,
    auto_save_interval_seconds=300
)
```

### 4. Query Optimization
Passe Query-Parameter für bessere Performance an:
```python
response = language_model.generate_response(
    query="...",
    context_size=3,  # Weniger Kontext = schneller
    min_activation=0.5  # Höherer Threshold = weniger Neuronen
)
```

## Beispiele

### Beispiel 1: Einfacher Chatbot

```python
def simple_chatbot():
    # Setup
    container = ApplicationContainer(Settings(database_path="chatbot.db"))
    container.initialize()
    
    language_model = LanguageModel(
        container.graph,
        container.compression_engine,
        container.query_engine,
        container.training_engine
    )
    
    # Load knowledge
    loader = PreTrainingLoader(language_model)
    loader.load_english_knowledge()
    
    # Chat loop
    print("Chatbot ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        response = language_model.generate_response(user_input)
        print(f"Bot: {response}")
    
    container.shutdown()
```

### Beispiel 2: Learning Assistant

```python
def learning_assistant():
    # Setup
    container = ApplicationContainer(Settings(database_path="assistant.db"))
    container.initialize()
    
    language_model = LanguageModel(
        container.graph,
        container.compression_engine,
        container.query_engine,
        container.training_engine
    )
    
    # Load base knowledge
    loader = PreTrainingLoader(language_model)
    loader.load_english_knowledge()
    
    # Teach new concepts
    concepts = [
        "Machine learning is a subset of AI that learns from data",
        "Neural networks are inspired by biological neurons",
        "Deep learning uses multiple layers of neural networks"
    ]
    
    for concept in concepts:
        language_model.learn(concept, tags=["AI", "ML"])
    
    # Ask questions
    questions = [
        "What is machine learning?",
        "Tell me about neural networks",
        "What is deep learning?"
    ]
    
    for question in questions:
        print(f"Q: {question}")
        response = language_model.generate_response(question)
        print(f"A: {response}\n")
    
    container.shutdown()
```

## Troubleshooting

### Problem: Keine Antworten

```python
# Lösung: Prüfe ob Wissen geladen wurde
stats = language_model.get_statistics()
if stats['knowledge_neurons'] == 0:
    loader = PreTrainingLoader(language_model)
    loader.load_english_knowledge()
```

### Problem: Schlechte Antworten

```python
# Lösung: Reduziere min_activation Threshold
response = language_model.generate_response(
    query="...",
    min_activation=0.1  # Niedrigerer Threshold
)
```

### Problem: Langsame Performance

```python
# Lösung: Reduziere context_size und propagation_depth
response = language_model.generate_response(
    query="...",
    context_size=3  # Weniger Kontext
)

results = language_model.understand(
    text="...",
    propagation_depth=2  # Weniger Tiefe
)
```

## Nächste Schritte

1. **Mehr Wissen hinzufügen**: Erstelle eigene Knowledge Bases
2. **Fine-tuning**: Trainiere die KI mit spezifischem Wissen
3. **Integration**: Integriere die KI in deine Anwendung
4. **Erweiterung**: Füge neue Neuron-Typen hinzu

## Siehe auch

- [Main Entry Point Documentation](./MAIN_ENTRY_POINT.md)
- [API Documentation](./api/README.md)
- [Custom Neuron Types](./CUSTOM_NEURON_TYPES.md)
