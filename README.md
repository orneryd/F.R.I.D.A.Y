# 🤖 F.R.I.D.A.Y AI

**An intelligent AI assistant based on a 3D neuron system**

F.R.I.D.A.Y (Friendly Responsive Intelligent Digital Assistant for You) is an advanced AI built on a unique 3D synaptic neuron system. The AI learns incrementally, automatically avoids duplicates, and offers natural conversation capabilities.

## ✨ Features

- 🧠 **3D Neuron Architecture**: Unique spatial neuron system
- 🤖 **Neural Inference Engine**: Real transformer logic (like GPT/BERT) for intelligent responses
- 💬 **Natural Conversation**: Over 300+ conversation patterns
- ⚡ **Incremental Training**: Fast updates without complete retraining
- 🗑️ **Automatic Duplicate Detection**: Keeps knowledge base clean
- 🎯 **Semantic Search**: Finds relevant information through similarity
- 💾 **Persistent Storage**: SQLite database for permanent learning
- 🚀 **GPU Acceleration**: 10-100x faster with CUDA/MPS support
- 🎮 **Simple CLI**: User-friendly command-line interface

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/F.R.I.D.A.Y.git
cd F.R.I.D.A.Y

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (NVIDIA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Check GPU status
python cli.py gpu-info
```

**⚠️ Important:** PyTorch requires **Python 3.8-3.12** (not 3.13!).

### First Steps

```bash
# 1. Train the AI (first time)
python cli.py train

# 2. Chat with the AI
python cli.py chat
```

### 🆕 Neural Inference Engine (NEU!)

Nutze echte Transformer-Logik für 20-40% bessere Antworten:

```bash
# Setup (einmalig)
python scripts/setup_neural_inference.py

# Demo testen
python examples/neural_inference_demo.py
```

**Vorteile:**
- ✅ Kontextuelles Verständnis statt nur Wort-Ähnlichkeit
- ✅ Multi-Head Attention wie in GPT/BERT
- ✅ Nutzt vortrainierte Hugging Face Modelle
- ✅ Dein Training-System bleibt gleich!

📖 **Mehr Info:** [Neural Inference Quick Start](docs/NEURAL_INFERENCE_QUICKSTART.md)

### 🚀 Dimension Upgrade (EMPFOHLEN!)

Upgrade zu **768 Dimensionen** für +28% bessere Qualität:

```bash
# Upgrade zu höheren Dimensionen
python scripts/migrate_to_higher_dimensions.py
```

**Warum upgraden?**
- ✅ 384D → 768D = **2x mehr Informationsdichte**
- ✅ Deutlich besseres kontextuelles Verständnis
- ✅ Präzisere und detailliertere Antworten
- ✅ Alte Datenbank bleibt erhalten

📖 **Mehr Info:** [Dimension Upgrade Guide](docs/DIMENSION_UPGRADE.md)

Das war's! Die KI ist jetzt einsatzbereit. 🎉

## 📖 Verwendung

### Basis-Befehle

```bash
# Vollständiges Training
python cli.py train

# Inkrementelles Update (schnell!)
python cli.py update

# Interaktiver Chat
python cli.py chat

# KI testen
python cli.py test

# Statistiken anzeigen
python cli.py stats
```

### Erweiterte Optionen

```bash
# Training mit externen Datasets
python cli.py train --with-datasets --max-samples 5000

# Chat mit mehr Kontext
python cli.py chat --context-size 10 --min-activation 0.3

# Eigene Datenbank verwenden
python cli.py train --database my_ai.db
python cli.py chat --database my_ai.db
```

Siehe [CLI_GUIDE.md](CLI_GUIDE.md) für detaillierte Dokumentation.

## 💬 Chat-Beispiel

```
You: Hello
AI: Hello! How can I help you today?

You: What are you?
AI: I'm an AI assistant designed to help answer questions

You: Can you learn?
AI: Yes, in this system I can learn and adapt based on interactions

You: What is AI?
AI: AI (Artificial Intelligence) is technology that enables machines 
    to perform tasks that typically require human intelligence

You: Thank you
AI: You're very welcome!
```

## 🧠 Architektur

### 3D-Neuronen-System

F.R.I.D.A.Y verwendet ein einzigartiges 3D-räumliches Neuronen-System:

- **Neuronen**: Wissenseinheiten im 3D-Raum positioniert
- **Synapsen**: Gewichtete Verbindungen zwischen verwandten Neuronen
- **Vektoren**: 384-dimensionale Embeddings für semantische Ähnlichkeit
- **Aktivierung**: Propagierung durch das Netzwerk für Kontext

### Komponenten

```
F.R.I.D.A.Y/
├── neuron_system/          # Kern-System
│   ├── core/               # Basis-Komponenten (Neuronen, Synapsen, Graph)
│   ├── engines/            # Verarbeitungs-Engines (Query, Training, Compression)
│   ├── ai/                 # KI-Komponenten
│   │   ├── language_model.py        # Haupt-Sprachmodell
│   │   ├── incremental_trainer.py   # Inkrementelles Training
│   │   ├── conversation_knowledge.py # Konversationsdaten
│   │   └── natural_dialogue.py      # Natürliche Dialoge
│   └── storage/            # Persistenz (SQLite)
├── cli.py                  # Kommandozeilen-Interface
└── main.py                 # Haupt-Container
```

## 📊 Performance

### Aktuelle Statistiken

- **Neuronen**: ~14,000
- **Synapsen**: ~8,500
- **Konnektivität**: 0.59 (gut vernetzt)
- **Datenbank-Größe**: ~50 MB
- **Antwortzeit**: < 1 Sekunde

### Trainingszeiten

- **Vollständiges Training**: 5-10 Minuten
- **Mit Datasets**: 20-30 Minuten
- **Inkrementelles Update**: 1-3 Minuten

## 🔧 Eigene Wissensbasis

### Neue Konversationen hinzufügen

Bearbeite `neuron_system/ai/conversation_knowledge.py`:

```python
DIRECT_QA = [
    "Question: Deine Frage? Answer: Deine Antwort",
    "Question: Wie geht es dir? Answer: Mir geht es gut!",
    # ... mehr Q&A-Paare
]
```

Dann Update ausführen:

```bash
python cli.py update
```

### Mehrsprachigkeit

F.R.I.D.A.Y unterstützt mehrere Sprachen:

```python
# Deutsch
"Question: Was bist du? Answer: Ich bin ein KI-Assistent",

# Englisch
"Question: What are you? Answer: I'm an AI assistant",

# Französisch
"Question: Qu'est-ce que tu es? Answer: Je suis un assistant IA",
```

## 🎯 Use Cases

### 1. Persönlicher Assistent
```bash
python cli.py chat
```
Stelle Fragen, erhalte Informationen, lerne neue Dinge.

### 2. Wissensdatenbank
```bash
# Eigenes Wissen hinzufügen
python cli.py update
```
Baue deine eigene spezialisierte Wissensbasis auf.

### 3. Chatbot-Backend
```python
from neuron_system.ai.language_model import LanguageModel

# In deiner Anwendung
response = language_model.generate_response(user_input)
```

### 4. Forschung & Experimente
```bash
# Verschiedene Konfigurationen testen
python cli.py train --database experiment1.db
python cli.py train --database experiment2.db --with-datasets
```

## 📚 Dokumentation

### Basis
- [CLI Guide](CLI_GUIDE.md) - Detaillierte CLI-Dokumentation
- [Quick Start](QUICKSTART.md) - Schnelleinstieg

### GPU-Beschleunigung
- [Quick GPU Setup](QUICK_GPU_SETUP.md) - ⚡ Schnelle GPU-Installation
- [INSTALL_CUDA.md](INSTALL_CUDA.md) - Detaillierte CUDA-Installation
- [GPU Acceleration](GPU_ACCELERATION.md) - Vollständige GPU-Anleitung
- [GPU Setup Summary](GPU_SETUP_SUMMARY.md) - Setup-Zusammenfassung

### Training
- [Reddit Training](REDDIT_TRAINING.md) - Reddit-Dataset-Training
- [Continuous Learning](CONTINUOUS_LEARNING.md) - Kontinuierliches Lernen
- [Architecture](docs/ARCHITECTURE.md) - System-Architektur
- [API Reference](docs/API.md) - API-Dokumentation
- [Training Guide](docs/TRAINING.md) - Training-Strategien

## 🤝 Beitragen

Beiträge sind willkommen! Hier sind einige Möglichkeiten:

1. **Neue Konversationsdaten**: Füge Q&A-Paare hinzu
2. **Bug-Fixes**: Melde oder behebe Bugs
3. **Features**: Schlage neue Features vor
4. **Dokumentation**: Verbessere die Docs

```bash
# Fork das Repository
# Erstelle einen Branch
git checkout -b feature/neue-funktion

# Committe deine Änderungen
git commit -m "Füge neue Funktion hinzu"

# Push zum Branch
git push origin feature/neue-funktion

# Erstelle einen Pull Request
```

## 🐛 Bekannte Probleme

- Einige Fragen matchen noch nicht perfekt mit Q&A-Paaren
- Performance bei sehr großen Datenbanken (>100k Neuronen) kann langsamer werden
- Antwort-Synthese kann manchmal zu kurz sein

Siehe [Issues](https://github.com/yourusername/F.R.I.D.A.Y/issues) für aktuelle Probleme.

## 🗺️ Roadmap

### Version 1.1 (Geplant)
- [ ] Verbesserte Antwort-Synthese
- [ ] Multi-Turn-Konversationen mit Kontext
- [ ] Web-Interface
- [ ] REST API

### Version 1.2 (Geplant)
- [ ] Mehrsprachige Unterstützung (Deutsch, Französisch, Spanisch)
- [ ] Langzeit-Gedächtnis
- [ ] Personalisierung pro Benutzer
- [ ] Voice-Interface

### Version 2.0 (Zukunft)
- [ ] Verteiltes Training
- [ ] Cloud-Deployment
- [ ] Mobile Apps
- [ ] Plugin-System

## 📄 Lizenz

MIT License - siehe [LICENSE](LICENSE) für Details.

## 🙏 Danksagungen

- **Sentence Transformers**: Für die Embedding-Modelle
- **SQLite**: Für die robuste Datenbank
- **HuggingFace**: Für die Datasets
- **Community**: Für Feedback und Beiträge

## 📞 Kontakt

- **GitHub**: [yourusername/F.R.I.D.A.Y](https://github.com/yourusername/F.R.I.D.A.Y)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/yourusername/F.R.I.D.A.Y/issues)
- **Discussions**: [Community Discussions](https://github.com/yourusername/F.R.I.D.A.Y/discussions)

## ⭐ Star History

Wenn dir F.R.I.D.A.Y gefällt, gib dem Projekt einen Stern! ⭐

---

**Made with ❤️ and 🧠 by the F.R.I.D.A.Y Team**

*"Your friendly AI companion for everyday tasks"*


## 📚 Dokumentation

Das Projekt hat 4 Haupt-Dokumentationen:

- **[CLI.md](CLI.md)** - Vollständiger CLI Guide mit allen Befehlen
- **[FEATURES.md](FEATURES.md)** - Detaillierte Feature-Dokumentation
- **[CHANGELOG.md](CHANGELOG.md)** - Versions-Historie und Updates
- **[docs/](docs/)** - Zusätzliche Dokumentation

### Quick Links
- [Installation](#installation)
- [Erste Schritte](#erste-schritte)
- [CLI Befehle](CLI.md)
- [Alle Features](FEATURES.md)
- [Neural Inference](FEATURES.md#neural-inference-engine)
- [Training System](FEATURES.md#training-system)
- [API Documentation](FEATURES.md#api)

## 📁 Projekt-Struktur

```
F.R.I.D.A.Y/
├── scripts/              # Utility-Scripts (Setup, Migration, Tests)
├── data/                 # Datenbanken und Logs
├── docs/                 # Dokumentation
├── examples/             # Code-Beispiele
├── tests/                # Tests
├── neuron_system/        # Core Code
│   ├── ai/              # AI Module (Training, Models, Inference)
│   ├── core/            # Core Components (Neurons, Synapses, Graph)
│   ├── engines/         # Engines (Compression, Query, Training)
│   ├── storage/         # Persistence Layer
│   └── ...
├── README.md            # Diese Datei
├── CLI.md               # CLI Guide
├── FEATURES.md          # Feature-Dokumentation
├── CHANGELOG.md         # Versions-Historie
├── cli.py               # Command Line Interface
└── requirements.txt     # Dependencies
```
