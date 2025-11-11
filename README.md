# F.R.I.D.A.Y - AI Assistant with Neuron System

**Friday** is an AI assistant built on a unique 3D synaptic neuron system with reasoning capabilities and generative text synthesis.

## 🌟 Features

- **3D Neuron Network** - Knowledge stored in interconnected neurons
- **Chain-of-Thought Reasoning** - Transparent thinking process
- **Generative Synthesis** - Creates original responses using local LLMs
- **Self-Reflection** - Validates and improves responses
- **Continuous Learning** - Learns from conversations
- **Local & Private** - Runs completely on your machine

## 🚀 Quick Start

### Installation

**Option 1: UV (Recommended - 10x faster)**
```bash
# Install UV
pip install uv

# Clone repository
git clone https://github.com/yourusername/F.R.I.D.A.Y.git
cd F.R.I.D.A.Y

# Install dependencies (automatic venv creation)
uv sync

# Activate venv and run
uv run python cli.py view
```

**Option 2: pip (Traditional)**
```bash
# Clone repository
git clone https://github.com/yourusername/F.R.I.D.A.Y.git
cd F.R.I.D.A.Y

# Install dependencies
pip install -r requirements.txt

# Run
python cli.py view
```

See `UV_SETUP.md` for detailed UV installation guide.

### Usage

#### Chat with Friday

```bash
python cli.py chat
```

#### Train Friday

```bash
# Train from conversation dataset (3000 samples)
python train.py conversations --max-samples 3000

# Show training statistics
python train.py stats

# Get help
python train.py --help
```

#### View Friday's Brain in 3D

```bash
# Start high-performance 3D viewer (Three.js/WebGL)
python cli.py view

# Custom port
python cli.py view --port 5001
```

Open http://localhost:5001 in your browser to see the neural network in 3D!

## 📁 Project Structure

```
Friday/
├── cli.py                          # Main CLI interface
├── train.py                        # Unified training system
├── requirements.txt                # Dependencies
│
├── neuron_system/                  # Core system
│   ├── core/                       # Core components (neurons, graph, synapses)
│   ├── engines/                    # Processing engines
│   ├── storage/                    # Database storage
│   ├── ai/                         # AI capabilities
│   │   ├── language_model.py       # Main language model
│   │   ├── generative_synthesis.py # Text generation with LLMs
│   │   ├── chain_of_thought.py     # Reasoning system
│   │   └── self_reflection.py      # Self-validation
│   └── training/                   # Training system
│       └── training_manager.py     # Unified training manager
│
├── data/                           # Data files
│   └── neuron_system.db            # Neuron database
│
└── docs/                           # Documentation
    ├── ARCHITECTURE_SIMPLE_EN.md   # System architecture
    └── CHAIN_OF_THOUGHT.md         # Reasoning documentation
```

## 🧠 How It Works

### 1. Neuron System

Friday stores knowledge in a 3D network of neurons connected by synapses. Each neuron contains:
- **Knowledge** - Text, facts, or conversation patterns
- **Position** - 3D coordinates in semantic space
- **Connections** - Synapses to related neurons

### 2. Reasoning

Friday uses Chain-of-Thought reasoning to:
1. **Understand** the question
2. **Classify** the question type
3. **Plan** the response strategy
4. **Synthesize** the answer

### 3. Generative Synthesis

Instead of just returning stored text, Friday uses local LLMs to:
- **Reformulate** answers in its own words
- **Combine** knowledge from multiple neurons
- **Generate** natural, contextual responses

Supports:
- **Ollama** (recommended) - Fast, local, high-quality
- **HuggingFace** - Automatic fallback
- **Template-based** - Works without external models

## 🎯 Training

Friday can learn from various sources:

### Conversation Datasets

```bash
python train.py conversations --max-samples 5000
```

### Q&A Pairs

Create a JSON file with Q&A pairs:

```json
[
  {
    "question": "What is AI?",
    "answer": "AI is artificial intelligence..."
  }
]
```

Train:

```bash
python train.py qa your_qa_file.json
```

## 🔧 Configuration

### Using Ollama (Recommended)

For best results, install Ollama:

1. Download from https://ollama.ai
2. Install and run: `ollama pull qwen2.5:1.5b`
3. Friday will automatically detect and use it

See `INSTALL_OLLAMA.md` for details.

### Database Location

Default: `data/neuron_system.db`

Change with `--database` flag:

```bash
python cli.py chat --database /path/to/db
python train.py stats --database /path/to/db
```

## 📊 Statistics

Check Friday's knowledge:

```bash
python train.py stats
```

Output:
```
Database: data/neuron_system.db
Total neurons: 4571
```

## 🛠️ Development

### Architecture

Friday uses a modular architecture:

- **Core** - Neuron graph, synapses, 3D vectors
- **Engines** - Compression, query, training
- **Storage** - SQLite database with batch operations
- **AI** - Language model, reasoning, generation

### Experimental Features

Some features are still in development:

- **Assimilation System** - Advanced knowledge extraction (in `scripts/`)
- See `EXPERIMENTAL.md` for details

### Adding Features

1. **New training sources** - Extend `TrainingManager`
2. **New AI capabilities** - Add to `neuron_system/ai/`
3. **New commands** - Extend `cli.py` or `train.py`

## 📝 Documentation

- `docs/ARCHITECTURE_SIMPLE_EN.md` - System architecture
- `docs/CHAIN_OF_THOUGHT.md` - Reasoning system
- `INSTALL_OLLAMA.md` - Ollama setup guide
- `CLI.md` - CLI documentation

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Built with sentence-transformers for embeddings
- Uses Ollama/HuggingFace for text generation
- Inspired by biological neural networks

## 📧 Contact

For questions or feedback, open an issue on GitHub.

---

**Friday** - Your local AI assistant with a brain! 🧠✨
