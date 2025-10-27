# 🤖 F.R.I.D.A.Y AI

**An intelligent AI assistant based on a 3D neuron system**

F.R.I.D.A.Y (Friendly Responsive Intelligent Digital Assistant for You) is an advanced AI built on a unique 3D synaptic neuron system. The AI learns incrementally, automatically avoids duplicates, and offers natural conversation capabilities.

## ✨ Features

- 🧠 **3D Neuron Architecture**: Unique spatial neuron system
- 🤖 **Neural Inference Engine**: Real transformer logic (like GPT/BERT) for intelligent responses
- 💬 **Natural Conversation**: Over 300+ conversation patterns
- ⚡ **Incremental Training**: Fast updates without complete retraining
- 🗑️ **Automatic Duplicate Detection**: Keeps knowledge base clean
- 🎯 **Semantic Search**: Finds relevant information through similarity
- 💾 **Persistent Storage**: SQLite database for permanent learning
- 🚀 **GPU Acceleration**: 10-100x faster with CUDA/MPS support
- 🎮 **Simple CLI**: User-friendly command-line interface

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/F.R.I.D.A.Y.git
cd F.R.I.D.A.Y

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (NVIDIA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Check GPU status
python cli.py gpu-info
```

**⚠️ Important:** PyTorch requires **Python 3.8-3.12** (not 3.13!).

### First Steps

```bash
# 1. Train the AI (first time)
python cli.py train

# 2. Chat with the AI
python cli.py chat
```

### 🆕 Neural Inference Engine (NEW!)

Use real transformer logic for 20-40% better responses:

```bash
# Setup (one-time)
python scripts/setup_neural_inference.py

# Test demo
python examples/neural_inference_demo.py
```

**Benefits:**
- ✅ Contextual understanding instead of just word similarity
- ✅ Multi-Head Attention like in GPT/BERT
- ✅ Uses pretrained Hugging Face models
- ✅ Your training system stays the same!

📖 **More Info:** [Neural Inference Quick Start](docs/NEURAL_INFERENCE_QUICKSTART.md)

### 🚀 Dimension Upgrade (RECOMMENDED!)

Upgrade to **768 dimensions** for +28% better quality:

```bash
# Upgrade to higher dimensions
python scripts/migrate_to_higher_dimensions.py
```

**Why upgrade?**
- ✅ 384D → 768D = **2x more information density**
- ✅ Significantly better contextual understanding
- ✅ More precise and detailed responses
- ✅ Old database remains preserved

📖 **More Info:** [Dimension Upgrade Guide](docs/DIMENSION_UPGRADE.md)

That's it! The AI is now ready to use. 🎉

## 📖 Usage

### Basic Commands

```bash
# Full training
python cli.py train

# Interactive chat
python cli.py chat

# Add knowledge
python cli.py learn "AI is artificial intelligence" --tags ai

# Query the AI
python cli.py query "What is AI?"

# Show statistics
python cli.py stats
```

### Advanced Options

```bash
# Training with external datasets
python cli.py train --with-datasets --max-samples 5000

# Chat with more context
python cli.py chat --context-size 10 --neural

# Use custom database
python cli.py train --database my_ai.db
python cli.py chat --database my_ai.db
```

See [CLI.md](CLI.md) for detailed documentation.

## 💬 Chat Example

```
You: Hello
AI: Hello! How can I help you today?

You: What are you?
AI: I'm an AI assistant designed to help answer questions

You: Can you learn?
AI: Yes, in this system I can learn and adapt based on interactions

You: What is AI?
AI: AI (Artificial Intelligence) is technology that enables machines 
    to perform tasks that typically require human intelligence

You: Thank you
AI: You're very welcome!
```

## 🧠 Architecture

### 3D Neuron System

F.R.I.D.A.Y uses a unique 3D spatial neuron system:

- **Neurons**: Knowledge units positioned in 3D space
- **Synapses**: Weighted connections between related neurons
- **Vectors**: 384-dimensional embeddings for semantic similarity
- **Activation**: Propagation through the network for context

### Components

```
F.R.I.D.A.Y/
├── neuron_system/          # Core System
│   ├── core/               # Base Components (Neurons, Synapses, Graph)
│   ├── engines/            # Processing Engines (Query, Training, Compression)
│   ├── ai/                 # AI Components
│   │   ├── language_model.py        # Main Language Model
│   │   ├── incremental_trainer.py   # Incremental Training
│   │   ├── conversation_knowledge.py # Conversation Data
│   │   └── natural_dialogue.py      # Natural Dialogues
│   └── storage/            # Persistence (SQLite)
├── cli.py                  # Command Line Interface
└── main.py                 # Main Container
```

## 📊 Performance

### Current Statistics

- **Neurons**: ~14,000
- **Synapses**: ~8,500
- **Connectivity**: 0.59 (well connected)
- **Database Size**: ~50 MB
- **Response Time**: < 1 second

### Training Times

- **Full Training**: 5-10 minutes
- **With Datasets**: 20-30 minutes
- **Incremental Update**: 1-3 minutes

## 🔧 Custom Knowledge Base

### Adding New Conversations

Edit `neuron_system/ai/conversation_knowledge.py`:

```python
DIRECT_QA = [
    "Question: Your question? Answer: Your answer",
    "Question: How are you? Answer: I'm doing well!",
    # ... more Q&A pairs
]
```

Then run update:

```bash
python cli.py learn "Your new knowledge" --tags category
```

### Multilingual Support

F.R.I.D.A.Y supports multiple languages:

```python
# German
"Question: Was bist du? Answer: Ich bin ein KI-Assistent",

# English
"Question: What are you? Answer: I'm an AI assistant",

# French
"Question: Qu'est-ce que tu es? Answer: Je suis un assistant IA",
```

## 🎯 Use Cases

### 1. Personal Assistant
```bash
python cli.py chat
```
Ask questions, get information, learn new things.

### 2. Knowledge Database
```bash
# Add your own knowledge
python cli.py learn "Your knowledge" --tags category
```
Build your own specialized knowledge base.

### 3. Chatbot Backend
```python
from neuron_system.ai.language_model import LanguageModel

# In your application
response = language_model.generate_response(user_input)
```

### 4. Research & Experiments
```bash
# Test different configurations
python cli.py train --database experiment1.db
python cli.py train --database experiment2.db --with-datasets
```

## 📚 Documentation

### Basics
- [CLI Guide](CLI.md) - Detailed CLI documentation
- [Features](FEATURES.md) - Complete feature documentation
- [Changelog](CHANGELOG.md) - Version history

### GPU Acceleration
- [Quick GPU Setup](QUICK_GPU_SETUP.md) - ⚡ Fast GPU installation
- [INSTALL_CUDA.md](INSTALL_CUDA.md) - Detailed CUDA installation
- [GPU Acceleration](GPU_ACCELERATION.md) - Complete GPU guide
- [GPU Setup Summary](GPU_SETUP_SUMMARY.md) - Setup summary

### Training
- [Reddit Training](REDDIT_TRAINING.md) - Reddit dataset training
- [Continuous Learning](CONTINUOUS_LEARNING.md) - Continuous learning
- [Architecture](docs/ARCHITECTURE.md) - System architecture
- [API Reference](docs/API.md) - API documentation
- [Training Guide](docs/TRAINING.md) - Training strategies

## 🤝 Contributing

Contributions are welcome! Here are some ways to contribute:

1. **New Conversation Data**: Add Q&A pairs
2. **Bug Fixes**: Report or fix bugs
3. **Features**: Propose new features
4. **Documentation**: Improve the docs

```bash
# Fork the repository
# Create a branch
git checkout -b feature/new-feature

# Commit your changes
git commit -m "Add new feature"

# Push to branch
git push origin feature/new-feature

# Create a Pull Request
```

## 🐛 Known Issues

- Some questions don't match perfectly with Q&A pairs yet
- Performance with very large databases (>100k neurons) can be slower
- Response synthesis can sometimes be too short

See [Issues](https://github.com/yourusername/F.R.I.D.A.Y/issues) for current issues.

## 🗺️ Roadmap

### Version 1.1 (Planned)
- [ ] Improved response synthesis
- [ ] Multi-turn conversations with context
- [ ] Web interface
- [ ] REST API

### Version 1.2 (Planned)
- [ ] Multilingual support (German, French, Spanish)
- [ ] Long-term memory
- [ ] Per-user personalization
- [ ] Voice interface

### Version 2.0 (Future)
- [ ] Distributed training
- [ ] Cloud deployment
- [ ] Mobile apps
- [ ] Plugin system

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Sentence Transformers**: For the embedding models
- **SQLite**: For the robust database
- **HuggingFace**: For the datasets
- **Community**: For feedback and contributions

## 📞 Contact

- **GitHub**: [yourusername/F.R.I.D.A.Y](https://github.com/yourusername/F.R.I.D.A.Y)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/yourusername/F.R.I.D.A.Y/issues)
- **Discussions**: [Community Discussions](https://github.com/yourusername/F.R.I.D.A.Y/discussions)

## ⭐ Star History

If you like F.R.I.D.A.Y, give the project a star! ⭐

---

**Made with ❤️ and 🧠 by the F.R.I.D.A.Y Team**

*"Your friendly AI companion for everyday tasks"*

## 📚 Documentation Overview

The project has 4 main documentation files:

- **[CLI.md](CLI.md)** - Complete CLI guide with all commands
- **[FEATURES.md](FEATURES.md)** - Detailed feature documentation
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and updates
- **[docs/](docs/)** - Additional documentation

### Quick Links
- [Installation](#installation)
- [First Steps](#first-steps)
- [CLI Commands](CLI.md)
- [All Features](FEATURES.md)
- [Neural Inference](FEATURES.md#neural-inference-engine)
- [Training System](FEATURES.md#training-system)
- [API Documentation](FEATURES.md#api)

## 📁 Project Structure

```
F.R.I.D.A.Y/
├── scripts/              # Utility scripts (Setup, Migration, Tests)
├── data/                 # Databases and logs
├── docs/                 # Documentation
├── examples/             # Code examples
├── tests/                # Tests
├── neuron_system/        # Core code
│   ├── ai/              # AI modules (Training, Models, Inference)
│   ├── core/            # Core components (Neurons, Synapses, Graph)
│   ├── engines/         # Engines (Compression, Query, Training)
│   ├── storage/         # Persistence layer
│   └── spatial/         # 3D spatial system
├── README.md            # This file
├── CLI.md               # CLI guide
├── FEATURES.md          # Feature documentation
├── CHANGELOG.md         # Version history
├── cli.py               # Command line interface
└── requirements.txt     # Dependencies
```

## 🎮 CLI Commands Overview

```bash
# Training & Learning
python cli.py train                    # Train the AI
python cli.py learn "text" --tags ai  # Add knowledge

# Interaction
python cli.py chat                     # Interactive chat
python cli.py query "What is AI?"      # Query the AI

# System
python cli.py stats                    # Show statistics
python cli.py gpu-info                 # GPU information
python cli.py list-datasets            # List available datasets

# Validation
python cli.py validate-persistence     # Test data persistence
python cli.py validate-3d              # Test 3D system

# Migration
python cli.py migrate old.db new.db    # Migrate database
```

See [CLI.md](CLI.md) for complete command documentation.
