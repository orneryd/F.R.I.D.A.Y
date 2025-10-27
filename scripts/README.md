# Scripts

Utility scripts für Setup, Testing und Maintenance.

## Setup Scripts

### `setup_neural_inference.py`
Installiert und testet Neural Inference Engine.
```bash
python scripts/setup_neural_inference.py
```

### `migrate_to_higher_dimensions.py`
Upgrade zu höheren Dimensionen (384D → 768D/1024D).
```bash
python scripts/migrate_to_higher_dimensions.py
```

## Testing Scripts

### `test_dynamic_dimensions.py`
Testet dynamische Dimensions-Erkennung.
```bash
python scripts/test_dynamic_dimensions.py
```

## Performance Scripts

### `benchmark_gpu.py`
GPU Performance Benchmarks.
```bash
python scripts/benchmark_gpu.py
```

## Hinweis

Alle Scripts können vom Root-Verzeichnis aus ausgeführt werden:
```bash
python scripts/script_name.py
```
