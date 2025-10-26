# Training Quickstart

## Schnellstart: KI mit echten Daten trainieren

### 1. Installation

```bash
pip install datasets
```

### 2. Training

```python
from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel
from neuron_system.ai.dataset_loader import DatasetLoader

# System initialisieren
settings = Settings(database_path="my_trained_ai.db")
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

# Wikipedia laden (1000 Artikel)
loader.load_wikipedia(language="en", max_articles=1000)

# Verbindungen erstellen (speichert automatisch)
loader.create_connections_batch(top_k=5, save_interval=500)

# Explizit speichern
loader.save_to_database()

# Cleanup
container.shutdown()
```

### 3. Verwenden

```python
# Lade die trainierte AI
settings = Settings(database_path="my_trained_ai.db")
container = ApplicationContainer(settings)
container.initialize()

language_model = LanguageModel(
    container.graph,
    container.compression_engine,
    container.query_engine,
    container.training_engine
)

# Stelle Fragen
response = language_model.generate_response("What is Wikipedia?")
print(response)

container.shutdown()
```

### 4. Beispiele ausführen

```bash
# Training
python examples/example_dataset_training.py

# Laden und verwenden
python examples/example_load_trained_ai.py

# Testen
python examples/example_load_trained_ai.py --test
```

## Wichtige Punkte

✅ **Automatisches Speichern**: Neuronen und Synapsen werden während des Trainings automatisch gespeichert

✅ **Persistenz**: Alles wird in SQLite-Datenbank gespeichert

✅ **Wiederverwendbar**: Trainierte AI kann jederzeit geladen werden

✅ **Skalierbar**: Von 100 bis 100,000+ Neuronen

## Empfohlene Datasets

- **Wikipedia**: Enzyklopädisches Wissen
- **BookCorpus**: Literarisches Englisch  
- **C4 (Common Crawl)**: Web-Text
- **Eigene Daten**: Text-Dateien, JSON

## Mehr Informationen

- [Vollständige Dataset-Training Dokumentation](docs/DATASET_TRAINING.md)
- [AI Language Model Dokumentation](docs/AI_LANGUAGE_MODEL.md)
- [Main Entry Point Dokumentation](docs/MAIN_ENTRY_POINT.md)
