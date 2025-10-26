# 🤖 F.R.I.D.A.Y AI

**Ein intelligenter KI-Assistent basierend auf einem 3D-Neuronen-System**

F.R.I.D.A.Y (Friendly Responsive Intelligent Digital Assistant for You) ist eine fortschrittliche KI, die auf einem einzigartigen 3D-synaptischen Neuronen-System basiert. Die KI lernt inkrementell, vermeidet Duplikate automatisch und bietet natürliche Konversationsfähigkeiten.

## ✨ Features

- 🧠 **3D-Neuronen-Architektur**: Einzigartiges räumliches Neuronen-System
- 💬 **Natürliche Konversation**: Über 300+ Konversationsmuster
- ⚡ **Inkrementelles Training**: Schnelle Updates ohne komplettes Neutraining
- 🗑️ **Automatische Duplikat-Erkennung**: Hält die Wissensbasis sauber
- 🎯 **Semantische Suche**: Findet relevante Informationen durch Ähnlichkeit
- 💾 **Persistente Speicherung**: SQLite-Datenbank für dauerhaftes Lernen
- 🚀 **GPU-Beschleunigung**: 10-100x schneller mit CUDA/MPS Support
- 🎮 **Einfache CLI**: Benutzerfreundliche Kommandozeilen-Befehle

## 🚀 Schnellstart

### Installation

```bash
# Repository klonen
git clone https://github.com/yourusername/F.R.I.D.A.Y.git
cd F.R.I.D.A.Y

# Dependencies installieren
pip install -r requirements.txt

# Für GPU-Beschleunigung (NVIDIA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# GPU-Status prüfen
python cli.py gpu-info
```

**⚠️ Wichtig:** PyTorch benötigt **Python 3.8-3.12** (nicht 3.13!). 
- Schnelle GPU-Setup: [QUICK_GPU_SETUP.md](QUICK_GPU_SETUP.md)
- Detaillierte Anleitung: [INSTALL_CUDA.md](INSTALL_CUDA.md)
- Python 3.13 Problem: [PYTHON_VERSION_FIX.md](PYTHON_VERSION_FIX.md)

### Erste Schritte

```bash
# 1. KI trainieren (beim ersten Mal)
python cli.py train

# 2. Mit der KI chatten
python cli.py chat
```

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
