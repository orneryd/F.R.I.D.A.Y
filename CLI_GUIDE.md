# F.R.I.D.A.Y AI - CLI Guide

Einfache Kommandozeilen-Befehle für das Training und die Nutzung der KI.

## 🚀 Schnellstart

### 1. Erste Installation & Training

```bash
# Vollständiges Training (beim ersten Mal)
python cli.py train
```

Das erstellt eine neue KI mit allen Basis-Kenntnissen. Dauert ca. 5-10 Minuten.

### 2. Mit der KI chatten

```bash
# Interaktiver Chat starten
python cli.py chat
```

Dann einfach Fragen stellen:
```
You: Hello
AI: Hello! How can I help you today?

You: What are you?
AI: I'm an AI assistant designed to help answer questions

You: exit
AI: Goodbye! Have a great day!
```

### 3. KI aktualisieren (schnell!)

```bash
# Inkrementelles Update - nur neue Daten hinzufügen
python cli.py update
```

Viel schneller als komplettes Neutraining! Fügt nur neue/verbesserte Kenntnisse hinzu.

## 📋 Alle Befehle

### `train` - Vollständiges Training

Erstellt eine neue KI von Grund auf.

```bash
# Basis-Training (empfohlen für Start)
python cli.py train

# Mit externen Datasets (mehr Wissen, dauert länger)
python cli.py train --with-datasets --max-samples 5000

# Eigene Datenbank verwenden
python cli.py train --database my_ai.db
```

**Optionen:**
- `--with-datasets`: Lädt zusätzliche Datasets (Wikipedia, etc.)
- `--max-samples N`: Maximale Anzahl Samples aus Datasets
- `--database PATH`: Pfad zur Datenbank-Datei

**Dauer:** 5-30 Minuten (je nach Optionen)

---

### `update` - Inkrementelles Update

Aktualisiert eine bestehende KI mit neuen Kenntnissen.

```bash
# Schnelles Update
python cli.py update

# Mit eigener Datenbank
python cli.py update --database my_ai.db
```

**Vorteile:**
- ⚡ Viel schneller als `train`
- 🔄 Behält bestehende Neuronen
- 🗑️ Entfernt automatisch Duplikate
- ✨ Fügt nur neue/verbesserte Daten hinzu

**Dauer:** 1-3 Minuten

---

### `chat` - Interaktiver Chat

Startet einen interaktiven Chat mit der KI.

```bash
# Standard-Chat
python cli.py chat

# Mit mehr Kontext (detailliertere Antworten)
python cli.py chat --context-size 10

# Mit höherer Aktivierungs-Schwelle (präzisere Antworten)
python cli.py chat --min-activation 0.3
```

**Optionen:**
- `--context-size N`: Anzahl Neuronen für Kontext (default: 5)
- `--min-activation X`: Minimale Aktivierung (0.0-1.0, default: 0.1)

**Chat-Befehle:**
- `exit` oder `quit`: Chat beenden
- `stats`: Statistiken anzeigen

---

### `test` - KI testen

Testet die KI mit vordefinierten Fragen.

```bash
# Standard-Test
python cli.py test

# Mit eigener Datenbank
python cli.py test --database my_ai.db
```

Zeigt Antworten auf typische Fragen wie:
- "What are you?"
- "Can you help me?"
- "What is AI?"
- etc.

---

### `stats` - Statistiken anzeigen

Zeigt Informationen über die KI.

```bash
# Statistiken anzeigen
python cli.py stats
```

Zeigt:
- Anzahl Neuronen
- Anzahl Synapsen
- Durchschnittliche Konnektivität
- Datenbank-Größe

---

### `gpu-info` - GPU-Informationen

Zeigt GPU-Status und Performance-Informationen.

```bash
# GPU-Info anzeigen
python cli.py gpu-info

# Mit Performance-Test
python cli.py gpu-info --test
```

**Optionen:**
- `--test`: Performance-Test durchführen

Zeigt:
- GPU-Erkennung (CUDA/MPS/CPU)
- Device-Informationen
- Speicher-Informationen (bei GPU)
- Batch-Size-Empfehlungen
- Performance-Benchmarks (mit --test)
- Geschätzte Trainingszeiten

**Beispiel-Output:**
```
Device Name:        NVIDIA GeForce RTX 3090
Device Type:        CUDA
GPU Available:      ✓ YES
Total Memory:       24.00 GB
```

Nützlich um zu prüfen ob GPU-Beschleunigung verfügbar ist!

---

### `learn` - Self-Learning Training ⭐ NEU!

Verbessert die KI durch selbstständiges Lernen und Optimierung.

```bash
# Standard Self-Learning (3 Runden)
python cli.py learn --save

# Intensives Training (5 Runden)
python cli.py learn --rounds 5 --save

# Mit detaillierter Ausgabe
python cli.py learn --rounds 3 --verbose --save

# Mehr Fragen pro Runde
python cli.py learn --rounds 3 --questions 30 --save
```

**Optionen:**
- `--rounds N`: Anzahl Trainingsrunden (default: 3)
- `--questions N`: Fragen pro Runde (default: 20)
- `--save`: Änderungen speichern (WICHTIG!)
- `--verbose`: Detaillierte Ausgabe zeigen

**Was macht Self-Learning?**
- 🧹 Entfernt schlechte Neuronen (falsche Antworten)
- ✂️ Entfernt schwache Synapsen (schlechte Verbindungen)
- 💪 Verstärkt gute Neuronen (richtige Antworten)
- 🎯 Verbessert Antwortqualität automatisch
- 📊 Zeigt Verbesserungen in Echtzeit

**Empfehlung:**
Führe Self-Learning regelmäßig aus (z.B. nach jedem Update):
```bash
python cli.py update
python cli.py learn --rounds 3 --save
```

**Dauer:** 2-5 Minuten pro Runde

---

## 🎯 Typische Workflows

### Workflow 1: Erste Nutzung

```bash
# 1. KI trainieren
python cli.py train

# 2. Testen
python cli.py test

# 3. Chatten
python cli.py chat
```

### Workflow 2: KI verbessern

```bash
# 1. Neue Konversationsdaten in conversation_knowledge.py hinzufügen

# 2. Inkrementelles Update
python cli.py update

# 3. Self-Learning Training (NEU!)
python cli.py learn --rounds 3 --save

# 4. Testen
python cli.py test
```

### Workflow 3: Mehrere KI-Versionen

```bash
# Basis-Version
python cli.py train --database basic_ai.db

# Erweiterte Version mit Datasets
python cli.py train --database advanced_ai.db --with-datasets

# Mit verschiedenen Versionen chatten
python cli.py chat --database basic_ai.db
python cli.py chat --database advanced_ai.db
```

---

## 💡 Tipps & Tricks

### Bessere Chat-Antworten

```bash
# Mehr Kontext = detailliertere Antworten
python cli.py chat --context-size 10

# Höhere Schwelle = präzisere Antworten
python cli.py chat --min-activation 0.3
```

### Schnelles Experimentieren

```bash
# Kleine Test-KI erstellen (schnell)
python cli.py train --database test_ai.db

# Testen
python cli.py test --database test_ai.db

# Löschen wenn nicht gut
rm test_ai.db
```

### Performance-Optimierung

```bash
# Nur Basis-Wissen (schnell, klein)
python cli.py train

# Mit Datasets (langsamer, aber mehr Wissen)
python cli.py train --with-datasets --max-samples 1000
```

---

## 🔧 Erweiterte Nutzung

### Eigene Wissensbasis hinzufügen

1. Bearbeite `neuron_system/ai/conversation_knowledge.py`
2. Füge neue Q&A-Paare hinzu:
   ```python
   "Question: Deine Frage? Answer: Deine Antwort",
   ```
3. Update ausführen:
   ```bash
   python cli.py update
   ```

### Mehrere Sprachen

```python
# In conversation_knowledge.py
"Question: Wie geht es dir? Answer: Mir geht es gut, danke!",
"Question: Was bist du? Answer: Ich bin ein KI-Assistent",
```

Dann:
```bash
python cli.py update
python cli.py chat
```

---

## 📊 Beispiel-Output

### Training
```
======================================================================
FULL AI TRAINING
======================================================================

Starting full training...

1. Loading built-in knowledge...
   [OK] Loaded 80 items
2. Loading conversation knowledge...
   [OK] Loaded conversation knowledge
3. Creating connections...
   [OK] Connections created
4. Saving to database...
   [OK] Saved

======================================================================
TRAINING COMPLETE!
======================================================================
Neurons: 3575
Synapses: 1943
Connectivity: 0.54
Database: comprehensive_ai.db
```

### Chat
```
======================================================================
F.R.I.D.A.Y AI - Interactive Chat
======================================================================

Type 'exit' or 'quit' to end the conversation

You: Hello
AI: Hello! How can I help you today?

You: What can you do?
AI: I can answer questions, provide information, and help with various tasks

You: exit
AI: Goodbye! Have a great day!
```

---

## ❓ Häufige Fragen

**Q: Wie lange dauert das Training?**
A: Basis-Training: 5-10 Min, Mit Datasets: 20-30 Min, Update: 1-3 Min

**Q: Wie groß wird die Datenbank?**
A: Basis: ~50-100 MB, Mit Datasets: ~200-500 MB

**Q: Kann ich mehrere KIs haben?**
A: Ja! Nutze `--database` für verschiedene Versionen.

**Q: Wie verbessere ich die Antworten?**
A: Füge mehr Q&A-Paare in `conversation_knowledge.py` hinzu und führe `update` aus.

**Q: Kann ich die KI offline nutzen?**
A: Ja! Nach dem Training funktioniert alles offline.

---

## 🆘 Hilfe

```bash
# Allgemeine Hilfe
python cli.py --help

# Hilfe für spezifischen Befehl
python cli.py train --help
python cli.py chat --help
```

---

## 🎉 Los geht's!

```bash
# Starte jetzt:
python cli.py train
python cli.py chat
```

Viel Spaß mit deiner KI! 🚀
