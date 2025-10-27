# Tests

Konsolidierte Test-Suite für F.R.I.D.A.Y AI.

## Test-Struktur

### Core Tests
- `test_core.py` - Core functionality (Neurons, Synapses, Graph)
- `test_engines.py` - Engines (Compression, Query, Training)
- `test_storage.py` - Persistence & Database

### AI Tests
- `test_ai.py` - AI functionality (Language Model, Training)
- `test_inference.py` - Neural Inference Engine

### API Tests
- `test_api.py` - REST API endpoints
- `test_sdk.py` - Python SDK

### Integration Tests
- `test_integration.py` - End-to-end workflows

## Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_core.py

# Specific test
pytest tests/test_core.py::test_neuron_creation

# With coverage
pytest tests/ --cov=neuron_system
```

## Test Categories

### Unit Tests
Test einzelne Komponenten isoliert.

### Integration Tests
Test Zusammenspiel mehrerer Komponenten.

### Performance Tests
Test Performance und GPU-Acceleration.

## Writing Tests

```python
import pytest
from neuron_system.core import Neuron, NeuronGraph

def test_neuron_creation():
    """Test neuron creation."""
    neuron = Neuron()
    assert neuron.id is not None
    assert neuron.position is not None
```

## Deprecated Tests

Alte Test-Files wurden archiviert in `tests/archive/`.
Nur aktuelle, wartbare Tests sind im Haupt-Ordner.
