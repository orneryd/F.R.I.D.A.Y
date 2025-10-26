# F.R.I.D.A.Y AI - Quick Command Reference

## 🚀 Essential Commands

### First Time Setup
```bash
# Train the AI (first time only)
python cli.py train
```

### Daily Usage
```bash
# Chat with AI
python cli.py chat

# Test AI responses
python cli.py test

# Show statistics
python cli.py stats
```

### Improving the AI
```bash
# Update with new knowledge (fast)
python cli.py update

# Self-learning training (optimize quality)
python cli.py learn --rounds 3 --save
```

---

## 📋 All Commands

| Command | Description | Duration |
|---------|-------------|----------|
| `train` | Full training from scratch | 5-30 min |
| `update` | Incremental update (add new knowledge) | 1-3 min |
| `learn` | Self-learning training (optimize quality) | 2-5 min/round |
| `chat` | Interactive chat with AI | - |
| `test` | Test with predefined questions | <1 min |
| `stats` | Show AI statistics | <1 sec |

---

## 🎯 Common Workflows

### Workflow 1: First Time
```bash
python cli.py train          # Train AI
python cli.py test           # Test it
python cli.py chat           # Chat with it
```

### Workflow 2: Add New Knowledge
```bash
# 1. Edit neuron_system/ai/conversation_knowledge.py
# 2. Add new Q&A pairs
python cli.py update         # Update AI
python cli.py learn --save   # Optimize quality
python cli.py test           # Test improvements
```

### Workflow 3: Optimize Quality
```bash
python cli.py learn --rounds 5 --save    # Intensive training
python cli.py test                        # Verify improvements
```

---

## ⚙️ Command Options

### `train` - Full Training
```bash
python cli.py train [options]

Options:
  --with-datasets          Include external datasets
  --max-samples N          Max samples from datasets (default: 5000)
  --database PATH          Database file path
```

### `update` - Incremental Update
```bash
python cli.py update [options]

Options:
  --database PATH          Database file path
```

### `learn` - Self-Learning ⭐ NEW!
```bash
python cli.py learn [options]

Options:
  --rounds N               Training rounds (default: 3)
  --questions N            Questions per round (default: 20)
  --save                   Save changes to database (REQUIRED!)
  --verbose                Show detailed output
```

### `chat` - Interactive Chat
```bash
python cli.py chat [options]

Options:
  --context-size N         Context neurons (default: 5)
  --min-activation X       Min activation threshold (default: 0.1)
  --database PATH          Database file path
```

### `test` - Test AI
```bash
python cli.py test [options]

Options:
  --database PATH          Database file path
```

### `stats` - Statistics
```bash
python cli.py stats [options]

Options:
  --database PATH          Database file path
```

---

## 💡 Pro Tips

### Better Responses
```bash
# More context = more detailed answers
python cli.py chat --context-size 10

# Higher threshold = more precise answers
python cli.py chat --min-activation 0.3
```

### Faster Training
```bash
# Quick training (no datasets)
python cli.py train

# With datasets (more knowledge, slower)
python cli.py train --with-datasets --max-samples 1000
```

### Quality Optimization
```bash
# Regular optimization (recommended)
python cli.py learn --rounds 3 --save

# Intensive optimization
python cli.py learn --rounds 10 --questions 30 --save
```

### Multiple AI Versions
```bash
# Create different versions
python cli.py train --database basic_ai.db
python cli.py train --database advanced_ai.db --with-datasets

# Use specific version
python cli.py chat --database basic_ai.db
python cli.py chat --database advanced_ai.db
```

---

## 🔧 Advanced Usage

### Custom Knowledge Base
```python
# Edit: neuron_system/ai/conversation_knowledge.py
CONVERSATION_KNOWLEDGE = [
    "Question: Your question? Answer: Your answer",
    # Add more...
]
```

Then:
```bash
python cli.py update
python cli.py learn --save
```

### Debugging
```bash
# Check database
python tests/check_db.py

# Count neurons by type
python tests/count_conversations.py
python tests/count_reasoning.py

# Debug vectors
python tests/debug_vectors.py
```

### Testing
```bash
# Run specific test
python tests/test_aggressive_training.py
python tests/test_training_improvement.py

# Run all tests
cd tests
python -m pytest
```

---

## 📊 Expected Results

### After Training
```
Neurons:  7,000-8,000
Synapses: 6,000-7,000
Database: 50-100 MB
```

### After Self-Learning (3 rounds)
```
Neurons removed:  30-40
Synapses removed: 150-200
Success rate:     20-40%
Quality:          Improved
```

---

## ❓ Quick Help

```bash
# General help
python cli.py --help

# Command-specific help
python cli.py train --help
python cli.py learn --help
python cli.py chat --help
```

---

## 🆘 Troubleshooting

### Database not found
```bash
# Create new database
python cli.py train
```

### Poor response quality
```bash
# Run self-learning
python cli.py learn --rounds 5 --save
```

### Slow performance
```bash
# Check database size
python cli.py stats

# Optimize with self-learning
python cli.py learn --save
```

### Want to start fresh
```bash
# Backup old database
mv comprehensive_ai.db comprehensive_ai.db.backup

# Create new one
python cli.py train
```

---

## 🎉 Quick Start

```bash
# 1. Train
python cli.py train

# 2. Optimize
python cli.py learn --rounds 3 --save

# 3. Chat
python cli.py chat
```

That's it! 🚀
