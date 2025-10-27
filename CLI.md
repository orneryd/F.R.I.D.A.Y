# F.R.I.D.A.Y AI - CLI Guide

Modern Command Line Interface with dimension support.

## Quick Start

```bash
# Training (384D)
python cli.py train

# Training (768D)
python cli.py --dimension 768 train

# Chat
python cli.py chat

# Chat with Neural Inference
python cli.py chat --neural
```

## Global Options

All commands support:

```bash
--database DB, -db DB      # Database path (default: data/neuron_system.db)
--dimension DIM, -d DIM    # Dimension: auto, 384, 768, 1024 (default: auto)
```

## Dimensions

The CLI supports different dimensions:

| Dimension | Model | Quality | Speed |
|-----------|-------|----------|-------|
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

## Commands

### `train` - Training

Trains the AI with base knowledge.

```bash
python cli.py train
python cli.py --dimension 768 train
python cli.py --database my_ai.db train
```

### `chat` - Interactive Chat

Starts interactive chat.

```bash
python cli.py chat
python cli.py chat --neural              # With Neural Inference
python cli.py chat --context-size 10     # More context
python cli.py --dimension 768 chat       # 768D system
```

**Options:**
- `--context-size N, -c N` - Number of context neurons (default: 5)
- `--neural, -n` - Use Neural Inference

**Chat commands:**
- `exit` or `quit` - Exit
- `stats` - Show statistics

### `learn` - Add Knowledge

Adds single knowledge item.

```bash
python cli.py learn "AI is artificial intelligence"
python cli.py learn "ML is machine learning" --tags ai,ml
python cli.py --dimension 768 learn "Deep learning info" --tags dl
```

**Options:**
- `--tags TAGS, -t TAGS` - Tags (comma-separated)

### `query` - Query

Queries the AI.

```bash
python cli.py query "What is AI?"
python cli.py query "Explain ML" --neural
python cli.py query "What is DL?" --top-k 10
```

**Options:**
- `--top-k N, -k N` - Number of results (default: 5)
- `--neural, -n` - Use Neural Inference

### `stats` - Statistics

Shows system statistics.

```bash
python cli.py stats
python cli.py --database my_ai.db stats
```

**Output:**
- Database path
- Dimension
- Number of neurons (total, by type)
- Number of synapses
- Average connectivity

### `gpu-info` - GPU Information

Shows GPU information.

```bash
python cli.py gpu-info
python cli.py gpu-info --test    # With performance test
```

**Output:**
- GPU name & type
- Availability
- Memory info

### `list-datasets` - List Datasets

Lists available training datasets.

```bash
python cli.py list-datasets
```

### `migrate` - Dimension Migration

Migrates data to different dimension.

```bash
# 384D → 768D
python cli.py --dimension 768 migrate old.db new_768d.db

# 768D → 1024D
python cli.py --dimension 1024 migrate old_768d.db new_1024d.db
```

**Process:**
1. Loads old database
2. Creates new database with new dimension
3. Re-compresses all neurons
4. Saves to new database

## Validation Commands

### `validate-persistence`

Validates that data is correctly saved and loaded.

```bash
python cli.py validate-persistence
```

**What is tested:**
- ✅ Neurons are correctly saved
- ✅ Synapses are correctly saved
- ✅ Data can be loaded
- ✅ System is functional after loading

**Output:**
```
Phase 1: Create and Save Data
  Creating 3 neurons...
  Before save: 3 neurons, 3 synapses
  ✓ Save successful

Phase 2: Load and Verify Data
  ✓ Load successful
  After load: 3 neurons, 3 synapses

Phase 3: Validation
  ✓ Neuron count matches: 3
  ✓ Synapse count matches: 3
  ✓ System functional after load

✅ Persistence validation PASSED
```

### `validate-3d`

Validates the 3D neuron system.

```bash
python cli.py validate-3d
```

**What is tested:**
- ✅ All neurons have 3D positions
- ✅ All neurons have vectors
- ✅ Positions are within bounds
- ✅ Synapses are valid
- ✅ Spatial index works
- ✅ Average connectivity

**Output:**
```
Creating test neurons for validation...
Created 5 test neurons

Neurons: 5
  With positions: 5
  With vectors: 5
  Within bounds: 5

Synapses: 10
  Valid: 10
  Avg connectivity: 2.00

Spatial index: Working (1 nearby neurons found)

✅ 3D System validation PASSED
```

**With different dimensions:**
```bash
# 384D
python cli.py validate-3d

# 768D
python cli.py --dimension 768 validate-3d

# 1024D
python cli.py --dimension 1024 validate-3d
```

## Workflows

### Workflow 1: First Steps (384D)

```bash
# 1. Training
python cli.py train

# 2. Test
python cli.py query "What is AI?"

# 3. Chat
python cli.py chat
```

### Workflow 2: High-Quality System (768D)

```bash
# 1. Training with 768D
python cli.py --dimension 768 --database ai_768d.db train

# 2. Add more knowledge
python cli.py --dimension 768 --database ai_768d.db learn "Advanced AI info" --tags ai,advanced

# 3. Chat with Neural Inference
python cli.py --dimension 768 --database ai_768d.db chat --neural
```

### Workflow 3: Migration

```bash
# 1. Existing 384D database
python cli.py --database old_384d.db stats

# 2. Migrate to 768D
python cli.py --dimension 768 migrate old_384d.db new_768d.db

# 3. Use
python cli.py --dimension 768 --database new_768d.db chat
```

## Tips & Tricks

### Performance

**Faster Training:**
```bash
# 384D is fastest
python cli.py train

# 768D is slower but better
python cli.py --dimension 768 train
```

**Faster Queries:**
```bash
# Less context
python cli.py chat --context-size 3

# Without Neural Inference
python cli.py chat
```

### Quality

**Better Responses:**
```bash
# Use 768D
python cli.py --dimension 768 chat

# Use Neural Inference
python cli.py chat --neural

# More context
python cli.py chat --context-size 10

# Combine all
python cli.py --dimension 768 chat --neural --context-size 10
```

### Choosing Dimension

**384D - For Testing & Development:**
```bash
python cli.py train
python cli.py chat
```

**768D - For Production (RECOMMENDED):**
```bash
python cli.py --dimension 768 --database prod.db train
python cli.py --dimension 768 --database prod.db chat --neural
```

**1024D - For Maximum Quality:**
```bash
python cli.py --dimension 1024 --database max.db train
python cli.py --dimension 1024 --database max.db chat --neural --context-size 10
```

## Troubleshooting

### "No GPU detected"
```bash
# Check GPU info
python cli.py gpu-info

# Continue with CPU (works automatically)
python cli.py train
```

### "Database locked"
```bash
# Use different database
python cli.py --database temp.db train
```

### "Out of memory"
```bash
# Use smaller dimension
python cli.py --dimension 384 train

# Or less context
python cli.py chat --context-size 3
```

### "Model not found"
```bash
# Models are downloaded automatically
# First time takes longer
# Make sure you have internet connection
```

---

**Version:** 2.2
**Last Updated:** October 2025
