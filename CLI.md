# F.R.I.D.A.Y AI - CLI Reference

Command Line Interface for the 3D Neuron AI System.

## Quick Start

```bash
python cli.py train                    # Train the AI
python cli.py chat                     # Start chatting
python cli.py stats                    # Show statistics
```

## Global Options

```bash
--database DB, -db DB      # Database path (default: data/neuron_system.db)
--dimension DIM, -d DIM    # Dimension: auto, 384, 768, 1024 (default: auto)
```

## Commands

### `train`
Train the AI with base knowledge.

```bash
python cli.py train
python cli.py --dimension 768 train
```

### `chat`
Interactive chat with the AI.

```bash
python cli.py chat
python cli.py chat --neural              # With Neural Inference
python cli.py chat --context-size 10     # More context
```

**Options:**
- `--context-size N, -c N` - Number of context neurons (default: 5)
- `--neural, -n` - Use Neural Inference

**Commands in chat:**
- `exit` or `quit` - Exit chat
- `stats` - Show statistics

### `learn`
Add single knowledge item.

```bash
python cli.py learn "AI is artificial intelligence"
python cli.py learn "ML is machine learning" --tags ai,ml
```

**Options:**
- `--tags TAGS, -t TAGS` - Tags (comma-separated)

### `query`
Query the AI.

```bash
python cli.py query "What is AI?"
python cli.py query "Explain ML" --neural
python cli.py query "What is DL?" --top-k 10
```

**Options:**
- `--top-k N, -k N` - Number of results (default: 5)
- `--neural, -n` - Use Neural Inference

### `stats`
Show system statistics.

```bash
python cli.py stats
```

**Shows:**
- Database path and dimension
- Number of neurons (total, by type)
- Number of synapses
- Average connectivity

### `gpu-info`
Show GPU information and status.

```bash
python cli.py gpu-info
python cli.py gpu-info --test    # With performance test
```

### `list-datasets`
List available training datasets.

```bash
python cli.py list-datasets
```

### `validate-persistence`
Validate data persistence (save/load).

```bash
python cli.py validate-persistence
```

**Tests:**
- Neurons are correctly saved
- Synapses are correctly saved
- Data can be loaded
- System is functional after loading

### `validate-3d`
Validate 3D neuron system.

```bash
python cli.py validate-3d
```

**Tests:**
- All neurons have 3D positions
- All neurons have vectors
- Positions are within bounds
- Synapses are valid
- Spatial index works
- Average connectivity

### `cluster`
Cluster neurons for better organization.

```bash
python cli.py cluster                           # Hybrid clustering (default)
python cli.py cluster --method kmeans --k 10    # K-Means clustering
python cli.py cluster --method dbscan           # Density-based clustering
python cli.py cluster --method topics           # Topic-based clustering
```

**Methods:**
- `kmeans` - K-Means clustering (vector-based)
- `dbscan` - DBSCAN clustering (density-based)
- `topics` - Topic-based clustering (tag-based)
- `hybrid` - Hybrid clustering (combines vectors + topics)

**Options:**
- `--method, -m` - Clustering method (default: hybrid)
- `--k` - Number of clusters for k-means/hybrid (default: 10)
- `--eps` - DBSCAN epsilon parameter (default: 0.3)
- `--min-samples` - DBSCAN min_samples (default: 3)

### `migrate`
Migrate database to different dimension.

```bash
python cli.py --dimension 768 migrate old.db new_768d.db
```

## Dimensions

| Dimension | Model | Quality | Speed |
|-----------|-------|---------|-------|
| 384 | all-MiniLM-L6-v2 | ⭐⭐⭐ | ⚡⚡⚡ |
| 768 | all-mpnet-base-v2 | ⭐⭐⭐⭐ | ⚡⚡ |
| 1024 | all-roberta-large-v1 | ⭐⭐⭐⭐⭐ | ⚡ |

**Examples:**
```bash
# 384D (Standard)
python cli.py train

# 768D (Recommended)
python cli.py --dimension 768 train

# 1024D (Maximum)
python cli.py --dimension 1024 train
```

## Common Workflows

### First Time Setup
```bash
python cli.py train                    # Train the AI
python cli.py query "What is AI?"      # Test it
python cli.py chat                     # Start chatting
```

### High Quality System (768D)
```bash
python cli.py --dimension 768 --database ai_768d.db train
python cli.py --dimension 768 --database ai_768d.db chat --neural
```

### Add Knowledge
```bash
python cli.py learn "Python is a programming language" --tags python,programming
python cli.py learn "React is a JavaScript library" --tags react,javascript
```

### Validate System
```bash
python cli.py validate-persistence     # Test persistence
python cli.py validate-3d              # Test 3D system
```

## Tips

**Better Quality:**
- Use 768D dimension
- Enable Neural Inference (`--neural`)
- Increase context size (`--context-size 10`)

**Faster Performance:**
- Use 384D dimension
- Disable Neural Inference
- Reduce context size (`--context-size 3`)

**Production Setup:**
```bash
python cli.py --dimension 768 --database prod.db train
python cli.py --dimension 768 --database prod.db chat --neural --context-size 10
```

---

**Version:** 2.2  
**Last Updated:** October 2025

For detailed documentation see [FEATURES.md](FEATURES.md) and [README.md](README.md).
