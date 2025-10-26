# Dataset Training Guide

## Überblick

Das Neuron-System kann mit echten Datasets trainiert werden, um umfangreiches Sprachwissen zu erwerben. Der `DatasetLoader` unterstützt verschiedene Datenquellen.

## Unterstützte Datenquellen

### 1. HuggingFace Datasets

Lade Datasets direkt von HuggingFace:

```python
from neuron_system.ai.dataset_loader import DatasetLoader

loader = DatasetLoader(language_model)

# Wikipedia
loader.load_wikipedia(language="en", max_articles=1000)

# BookCorpus
loader.load_bookcorpus(max_samples=1000)

# Common Crawl (C4)
loader.load_common_crawl_sample(max_samples=1000)

# Beliebiges HuggingFace Dataset
loader.load_from_huggingface(
    dataset_name="squad",
    split="train",
    text_field="context",
    max_samples=500
)
```

### 2. Text-Dateien

Lade Text aus lokalen Dateien:

```python
loader.load_from_text_file(
    file_path="my_text.txt",
    chunk_size=500,  # Größe der Text-Chunks
    overlap=50,      # Überlappung zwischen Chunks
    tags=["custom", "knowledge"]
)
```

### 3. JSON-Dateien

Lade strukturierte Daten aus JSON:

```python
# JSON Format:
# [
#   {"text": "...", "tags": ["tag1", "tag2"]},
#   {"text": "...", "tags": ["tag3"]}
# ]

loader.load_from_json(
    file_path="data.json",
    text_field="text",
    tags_field="tags",
    max_items=1000
)
```

## Installation

Für HuggingFace Datasets:

```bash
pip install datasets
```

## Schnellstart

### Einfaches Beispiel

```python
from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel
from neuron_system.ai.dataset_loader import DatasetLoader

# System initialisieren
settings = Settings(database_path="trained_ai.db")
container = ApplicationContainer(settings)
container.initialize()

# Language Model erstellen
language_model = LanguageModel(
    container.graph,
    container.compression_engine,
    container.query_engine,
    container.training_engine
)

# Dataset Loader erstellen
loader = DatasetLoader(language_model)

# Wikipedia laden (100 Artikel)
loader.load_wikipedia(language="en", max_articles=100)

# Verbindungen erstellen
loader.create_connections_batch(top_k=5)

# Statistiken anzeigen
stats = loader.get_statistics()
print(f"Loaded {stats['total_loaded']} items")
print(f"Created {stats['neurons_in_graph']} neurons")
print(f"Created {stats['synapses_in_graph']} synapses")

# Testen
response = language_model.generate_response("What is Wikipedia?")
print(response)

# Cleanup
container.shutdown()
```

### Vollständiges Beispiel

```bash
python examples/example_dataset_training.py
```

## Empfohlene Datasets

### Für Englisch

1. **Wikipedia** (Enzyklopädisches Wissen)
   ```python
   loader.load_wikipedia(language="en", max_articles=10000)
   ```

2. **BookCorpus** (Literarisches Englisch)
   ```python
   loader.load_bookcorpus(max_samples=5000)
   ```

3. **C4 (Common Crawl)** (Web-Text)
   ```python
   loader.load_common_crawl_sample(max_samples=5000)
   ```

4. **OpenWebText** (Reddit-Links)
   ```python
   loader.load_from_huggingface("openwebtext", max_samples=5000)
   ```

### Für Deutsch

```python
# Deutsche Wikipedia
loader.load_wikipedia(language="de", max_articles=5000)

# OSCAR (multilingual)
loader.load_from_huggingface(
    "oscar",
    split="train",
    text_field="text",
    max_samples=5000
)
```

### Für spezifische Domänen

```python
# Medizin
loader.load_from_huggingface("pubmed", max_samples=1000)

# Code
loader.load_from_huggingface("codeparrot/github-code", max_samples=1000)

# Wissenschaft
loader.load_from_huggingface("scientific_papers", max_samples=1000)
```

## Performance-Tipps

### 1. Batch-Loading

Lade Daten in Batches und erstelle Verbindungen am Ende:

```python
# Lade alle Daten OHNE Verbindungen
loader.load_wikipedia(max_articles=1000)
loader.load_bookcorpus(max_samples=500)

# Erstelle Verbindungen in einem Batch
loader.create_connections_batch(top_k=5)
```

### 2. Chunk-Größe optimieren

Für Text-Dateien:

```python
# Kleinere Chunks = mehr Neuronen, präziser
loader.load_from_text_file("file.txt", chunk_size=300, overlap=30)

# Größere Chunks = weniger Neuronen, schneller
loader.load_from_text_file("file.txt", chunk_size=1000, overlap=100)
```

### 3. Streaming für große Datasets

HuggingFace Datasets werden automatisch gestreamt:

```python
# Kein Problem, auch bei riesigen Datasets
loader.load_from_huggingface(
    "c4",
    max_samples=100000  # Lädt nur was gebraucht wird
)
```

### 4. Speicher-Management

```python
# Für sehr große Trainings:
settings = Settings(
    database_path="large_ai.db",
    enable_auto_save=True,
    auto_save_interval_seconds=300  # Speichere alle 5 Minuten
)
```

## Trainings-Strategien

### Strategie 1: Breites Wissen

Lade viele verschiedene Datasets für allgemeines Wissen:

```python
loader.load_wikipedia(max_articles=5000)
loader.load_bookcorpus(max_samples=2000)
loader.load_common_crawl_sample(max_samples=3000)
```

### Strategie 2: Domänen-Spezialist

Fokussiere auf eine spezifische Domäne:

```python
# Medizin-Spezialist
loader.load_from_huggingface("pubmed", max_samples=10000)
loader.load_from_huggingface("medical_questions", max_samples=5000)
```

### Strategie 3: Mehrsprachig

Trainiere mit mehreren Sprachen:

```python
loader.load_wikipedia(language="en", max_articles=3000)
loader.load_wikipedia(language="de", max_articles=3000)
loader.load_wikipedia(language="fr", max_articles=3000)
```

## Monitoring

### Progress Tracking

```python
# Batch-Size für häufigere Updates
loader.load_wikipedia(
    max_articles=10000,
    batch_size=10  # Log alle 10 Artikel
)
```

### Statistiken

```python
stats = loader.get_statistics()
print(f"Total loaded: {stats['total_loaded']}")
print(f"Neurons: {stats['neurons_in_graph']}")
print(f"Synapses: {stats['synapses_in_graph']}")

# Durchschnittliche Konnektivität
avg_conn = stats['synapses_in_graph'] / stats['neurons_in_graph']
print(f"Average connectivity: {avg_conn:.2f}")
```

## Troubleshooting

### Problem: "datasets library not installed"

```bash
pip install datasets
```

### Problem: Zu langsam

```python
# Reduziere max_samples
loader.load_wikipedia(max_articles=100)  # Statt 10000

# Erhöhe batch_size
loader.load_wikipedia(max_articles=1000, batch_size=100)

# Deaktiviere Verbindungen während des Ladens
loader.load_wikipedia(...)  # create_connections=False ist default
loader.create_connections_batch()  # Erstelle am Ende
```

### Problem: Zu viel Speicher

```python
# Verwende kleinere Samples
loader.load_wikipedia(max_articles=500)

# Aktiviere Auto-Save
settings = Settings(
    enable_auto_save=True,
    auto_save_interval_seconds=300
)
```

### Problem: Schlechte Antworten

```python
# Lade mehr Daten
loader.load_wikipedia(max_articles=5000)  # Mehr ist besser

# Erhöhe Konnektivität
loader.create_connections_batch(top_k=10)  # Mehr Verbindungen

# Passe Query-Parameter an
response = language_model.generate_response(
    query="...",
    context_size=5,  # Mehr Kontext
    min_activation=0.1  # Niedrigerer Threshold
)
```

## Beispiel-Workflows

### Workflow 1: Schneller Test

```python
# 5 Minuten Setup
loader.load_wikipedia(language="en", max_articles=100)
loader.create_connections_batch(top_k=3)
```

### Workflow 2: Produktions-Training

```python
# 1-2 Stunden Training
loader.load_wikipedia(language="en", max_articles=10000)
loader.load_bookcorpus(max_samples=5000)
loader.load_common_crawl_sample(max_samples=5000)
loader.create_connections_batch(top_k=5)
```

### Workflow 3: Umfassendes Training

```python
# Mehrere Stunden bis Tage
loader.load_wikipedia(language="en", max_articles=50000)
loader.load_bookcorpus(max_samples=20000)
loader.load_common_crawl_sample(max_samples=30000)
loader.load_from_huggingface("openwebtext", max_samples=20000)
loader.create_connections_batch(top_k=10)
```

## Speichern und Laden

### Automatisches Speichern

Das System speichert automatisch während des Trainings:

```python
# Speichert automatisch alle 500 Neuronen
loader.create_connections_batch(
    top_k=5,
    save_interval=500
)
```

### Explizites Speichern

```python
# Speichere alles explizit
loader.save_to_database()
```

### Trainierte AI laden

```python
# Lade eine bereits trainierte AI
settings = Settings(database_path="trained_ai.db")
container = ApplicationContainer(settings)
container.initialize()

# Die Neuronen und Synapsen werden automatisch geladen!
language_model = LanguageModel(
    container.graph,
    container.compression_engine,
    container.query_engine,
    container.training_engine
)

# Jetzt kannst du die trainierte AI verwenden
response = language_model.generate_response("What is Wikipedia?")
```

### Beispiel: Laden und Verwenden

```bash
# 1. Trainiere die AI
python examples/example_dataset_training.py

# 2. Lade die trainierte AI
python examples/example_load_trained_ai.py

# 3. Oder teste sie
python examples/example_load_trained_ai.py --test
```

### Datenbank-Verwaltung

```python
# Verschiedene Datenbanken für verschiedene Zwecke
settings_general = Settings(database_path="general_knowledge.db")
settings_medical = Settings(database_path="medical_knowledge.db")
settings_code = Settings(database_path="code_knowledge.db")

# Backup erstellen
import shutil
shutil.copy("trained_ai.db", "trained_ai_backup.db")
```

## Nächste Schritte

1. Starte mit einem kleinen Dataset (100-1000 Samples)
2. Teste die Antwortqualität
3. Erhöhe schrittweise die Datenmenge
4. Experimentiere mit verschiedenen Datasets
5. Optimiere die Konnektivität (top_k Parameter)
6. **Speichere die trainierte AI in einer Datenbank**
7. **Lade die trainierte AI für Produktion**

## Siehe auch

- [AI Language Model Documentation](./AI_LANGUAGE_MODEL.md)
- [Main Entry Point Documentation](./MAIN_ENTRY_POINT.md)
- [HuggingFace Datasets](https://huggingface.co/datasets)
