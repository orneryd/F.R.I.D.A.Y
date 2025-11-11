# Scripts Directory

Utility scripts for F.R.I.D.A.Y development and maintenance.

## Essential Scripts

### Setup & Configuration

#### `setup_model.py`
Download and setup local language models (Ollama/HuggingFace).

```bash
python scripts/setup_model.py
```

#### `setup_neural_inference.py`
Setup neural inference system with pretrained models.

```bash
python scripts/setup_neural_inference.py
```

### Maintenance & Repair

#### `reset_brain.py`
Reset the entire brain database (WARNING: Deletes all data).

```bash
python scripts/reset_brain.py
```

#### `reset_connections.py`
Reset only synapse connections, keep neurons intact.

```bash
python scripts/reset_connections.py
```

#### `fix_neuron_positions.py`
Fix and recalculate neuron 3D positions.

```bash
python scripts/fix_neuron_positions.py
```

### Data & Analysis

#### `extract_knowledge.py`
Extract knowledge from various sources into the brain.

```bash
python scripts/extract_knowledge.py --source data.json
```

#### `analyze_synapses.py`
Analyze synapse patterns and statistics.

```bash
python scripts/analyze_synapses.py
```

### Visualization

#### `live_viewer.py`
Live brain viewer with real-time updates (alternative to cli.py view).

```bash
python scripts/live_viewer.py --database data/neuron_system.db
```

#### `view_brain_optimized.py`
Optimized brain viewer for very large networks.

```bash
python scripts/view_brain_optimized.py --database data/neuron_system.db
```

### Performance & Testing

#### `benchmark_gpu.py`
Benchmark GPU performance for neural operations.

```bash
python scripts/benchmark_gpu.py
```

#### `demo_hebbian.py`
Demonstrate Hebbian learning in action.

```bash
python scripts/demo_hebbian.py
```

### Migration

#### `migrate_to_higher_dimensions.py`
Migrate database to higher dimensions (384D → 768D → 1024D).

```bash
python scripts/migrate_to_higher_dimensions.py
```

## Usage Notes

Most functionality is available through the main CLI:

```bash
python cli.py train          # Training
python cli.py stats          # Statistics
python cli.py view           # 3D Viewer
python cli.py chat           # Chat interface
python cli.py validate-3d    # Validate 3D system
python cli.py cluster        # Cluster neurons
```

Use scripts only for:
- Initial setup (setup_model.py, setup_neural_inference.py)
- Maintenance (reset_brain.py, fix_neuron_positions.py)
- Advanced operations (migrate_to_higher_dimensions.py)
- Development/testing (demo_hebbian.py, benchmark_gpu.py)

See `CLI.md` for complete CLI documentation.
