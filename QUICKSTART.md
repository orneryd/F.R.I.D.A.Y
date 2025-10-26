# 🚀 F.R.I.D.A.Y AI - Quickstart

**In 5 Minuten zur eigenen KI!**

## Schritt 1: Installation (2 Minuten)

```bash
# 1. Repository klonen oder herunterladen
git clone https://github.com/yourusername/F.R.I.D.A.Y.git
cd F.R.I.D.A.Y

# 2. Dependencies installieren
pip install -r requirements.txt
```

Das war's! Alle benötigten Pakete sind jetzt installiert.

## Schritt 2: KI trainieren (3 Minuten)

```bash
python cli.py train
```

Das Programm fragt:
```
Continue? (yes/no):
```

Tippe `yes` und drücke Enter.

Die KI wird jetzt trainiert. Du siehst:
```
1. Loading built-in knowledge...
   [OK] Loaded 80 items
2. Loading conversation knowledge...
   [OK] Loaded conversation knowledge
...
TRAINING COMPLETE!
Neurons: 3575
```

## Schritt 3: Mit der KI chatten! (sofort)

```bash
python cli.py chat
```

Jetzt kannst du mit der KI sprechen:

```
You: Hello
AI: Hello! How can I help you today?

You: What are you?
AI: I'm an AI assistant designed to help answer questions

You: What can you do?
AI: I can answer questions, provide information, and help with various tasks

You: Thank you
AI: You're very welcome!

You: exit
AI: Goodbye! Have a great day!
```

## 🎉 Fertig!

Du hast jetzt eine funktionierende KI!

## Was jetzt?

### Die KI testen
```bash
python cli.py test
```
Zeigt Antworten auf typische Fragen.

### Statistiken anzeigen
```bash
python cli.py stats
```
Zeigt wie viele Neuronen und Verbindungen die KI hat.

### Die KI verbessern
```bash
python cli.py update
```
Fügt neue Kenntnisse hinzu (sehr schnell!).

## Tipps für bessere Gespräche

### 1. Stelle klare Fragen
✅ Gut: "What is AI?"
❌ Schlecht: "Tell me everything about technology"

### 2. Sei spezifisch
✅ Gut: "Can you help me?"
❌ Schlecht: "I need something"

### 3. Nutze natürliche Sprache
✅ Gut: "How are you?"
✅ Gut: "What can you do?"
✅ Gut: "Thank you"

## Häufige Probleme

### "ModuleNotFoundError"
```bash
# Lösung: Dependencies installieren
pip install -r requirements.txt
```

### "Database not found"
```bash
# Lösung: Erst trainieren
python cli.py train
```

### KI antwortet nicht gut
```bash
# Lösung: Update ausführen
python cli.py update
```

## Nächste Schritte

1. **Lies die [CLI Guide](CLI_GUIDE.md)** für alle Befehle
2. **Lies die [README](README.md)** für Details
3. **Füge eigenes Wissen hinzu** in `conversation_knowledge.py`
4. **Experimentiere** mit verschiedenen Einstellungen

## Hilfe bekommen

- **Dokumentation**: Siehe [CLI_GUIDE.md](CLI_GUIDE.md)
- **Probleme**: Öffne ein [Issue](https://github.com/yourusername/F.R.I.D.A.Y/issues)
- **Fragen**: Siehe [Discussions](https://github.com/yourusername/F.R.I.D.A.Y/discussions)

## Viel Spaß! 🎉

```bash
python cli.py chat
```

Deine KI wartet auf dich! 🤖
