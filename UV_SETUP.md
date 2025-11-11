# UV Setup Guide

## Was ist UV?

UV ist ein extrem schneller Python Package Manager von Astral (Ruff-Entwickler). UV ist **10-100x schneller** als pip und hat besseres Dependency Management.

## Installation

### Windows
```powershell
# Mit PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Oder mit pip
pip install uv
```

### Linux/macOS
```bash
# Mit curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# Oder mit pip
pip install uv
```

**Wichtig:** F.R.I.D.A.Y benötigt Python 3.12+

```bash
# Python Version prüfen
python --version  # Sollte 3.12.x sein

# Falls nicht, Python 3.12 installieren:
# Windows: https://www.python.org/downloads/
# Linux: sudo apt install python3.12
# macOS: brew install python@3.12
```

## F.R.I.D.A.Y mit UV installieren

### 1. Repository klonen
```bash
git clone https://github.com/yourusername/F.R.I.D.A.Y.git
cd F.R.I.D.A.Y
```

### 2. Dependencies installieren
```bash
# Alle Dependencies installieren
uv sync

# Oder mit Dev-Dependencies
uv sync --all-extras
```

Das war's! UV erstellt automatisch ein Virtual Environment und installiert alle Dependencies.

### 3. Aktivieren und Starten
```bash
# Virtual Environment aktivieren
# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

# F.R.I.D.A.Y starten
python cli.py view
```

## UV Commands

### Sync (Dependencies installieren)
```bash
# Basis-Installation
uv sync

# Mit Dev-Dependencies
uv sync --dev

# Mit allen Extras (dev + viz)
uv sync --all-extras

# Nur bestimmte Extras
uv sync --extra viz
```

### Add (Neue Dependency hinzufügen)
```bash
# Neue Dependency hinzufügen
uv add requests

# Dev-Dependency hinzufügen
uv add --dev pytest

# Mit Version
uv add "numpy>=1.24.0"
```

### Remove (Dependency entfernen)
```bash
uv remove requests
```

### Update (Dependencies aktualisieren)
```bash
# Alle Dependencies aktualisieren
uv sync --upgrade

# Bestimmte Dependency aktualisieren
uv add --upgrade numpy
```

### Run (Command im venv ausführen)
```bash
# Python-Script ausführen
uv run python cli.py view

# Pytest ausführen
uv run pytest

# Direkt ohne Aktivierung
uv run friday view
```

### Lock (Lockfile erstellen)
```bash
# Lockfile aktualisieren
uv lock

# Lockfile mit neuen Dependencies
uv lock --upgrade
```

## Vorteile von UV

### 1. Geschwindigkeit
```
pip install:     ~60 Sekunden
uv sync:         ~3 Sekunden
```

### 2. Besseres Dependency Management
- Automatische Konflikt-Auflösung
- Lockfile für reproduzierbare Builds
- Parallele Downloads

### 3. Virtual Environment Management
- Automatische venv-Erstellung
- Keine manuellen Schritte nötig
- Saubere Isolation

### 4. Kompatibilität
- 100% kompatibel mit pip
- Nutzt PyPI
- Funktioniert mit requirements.txt

## PyTorch mit CUDA

UV ist so konfiguriert, dass es automatisch PyTorch mit CUDA 12.1 Support installiert:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu121" }
torchvision = { index = "pytorch-cu121" }
torchaudio = { index = "pytorch-cu121" }
```

Falls du CPU-only willst:
```bash
# pyproject.toml bearbeiten und [tool.uv.sources] entfernen
# Dann:
uv sync
```

## Workflow-Beispiele

### Neues Projekt Setup
```bash
git clone https://github.com/yourusername/F.R.I.D.A.Y.git
cd F.R.I.D.A.Y
uv sync
uv run python cli.py view
```

### Development
```bash
# Dev-Dependencies installieren
uv sync --dev

# Tests ausführen
uv run pytest

# Code formatieren
uv run black .

# Linting
uv run flake8
```

### Neue Dependency hinzufügen
```bash
# Dependency hinzufügen
uv add flask-cors

# Testen
uv run python cli.py view

# Committen
git add pyproject.toml uv.lock
git commit -m "Add flask-cors"
```

### Update auf neue Version
```bash
# Alle Dependencies aktualisieren
uv sync --upgrade

# Testen
uv run pytest

# Committen
git add uv.lock
git commit -m "Update dependencies"
```

## Troubleshooting

### "uv: command not found"
```bash
# UV neu installieren
pip install uv

# Oder PATH prüfen
echo $PATH  # Linux/macOS
echo $env:PATH  # Windows
```

### "Failed to download package"
```bash
# Cache löschen
uv cache clean

# Neu versuchen
uv sync
```

### "Dependency conflict"
```bash
# Lockfile neu erstellen
rm uv.lock
uv lock

# Sync
uv sync
```

### "PyTorch CUDA not found"
```bash
# Prüfe ob CUDA-Version korrekt ist
python -c "import torch; print(torch.cuda.is_available())"

# Falls nicht, manuell installieren:
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Migration von pip zu UV

### 1. Bestehende venv löschen
```bash
# Windows
rmdir /s .venv

# Linux/macOS
rm -rf .venv
```

### 2. UV installieren
```bash
pip install uv
```

### 3. Dependencies installieren
```bash
uv sync
```

### 4. Testen
```bash
uv run python cli.py view
```

## Vergleich: pip vs UV

| Feature | pip | UV |
|---------|-----|-----|
| Installation Zeit | ~60s | ~3s |
| Dependency Resolution | Langsam | Sehr schnell |
| Lockfile | Nein | Ja (uv.lock) |
| Virtual Env | Manuell | Automatisch |
| Parallel Downloads | Nein | Ja |
| Cache | Basic | Intelligent |

## Best Practices

### 1. Immer uv.lock committen
```bash
git add uv.lock
git commit -m "Update lockfile"
```

### 2. Regelmäßig updaten
```bash
# Wöchentlich
uv sync --upgrade
```

### 3. Dev-Dependencies trennen
```bash
# Nur für Development
uv add --dev pytest black flake8
```

### 4. Extras nutzen
```bash
# Nur was du brauchst
uv sync --extra viz  # Nur Visualisierung
```

## Links

- **UV Docs**: https://docs.astral.sh/uv/
- **UV GitHub**: https://github.com/astral-sh/uv
- **PyPI**: https://pypi.org/project/uv/

---

**Version:** 2.3
**UV Version:** 0.1.0+
**Status:** Produktionsbereit
