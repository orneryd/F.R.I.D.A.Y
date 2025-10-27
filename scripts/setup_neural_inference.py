"""
Setup Script für Neural Inference Engine.

Installiert alle benötigten Dependencies und testet die Installation.
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def check_package(package_name):
    """Check if a package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def install_package(package):
    """Install a package using pip."""
    logger.info(f"Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        logger.error(f"Failed to install {package}")
        return False


def main():
    """Main setup function."""
    
    print("\n" + "=" * 70)
    print("NEURAL INFERENCE ENGINE SETUP")
    print("=" * 70 + "\n")
    
    # Check Python version
    print("Step 1: Checking Python version")
    print("-" * 70)
    
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Python 3.8 or higher required!")
        return False
    
    print("✓ Python version OK\n")
    
    # Check required packages
    print("Step 2: Checking required packages")
    print("-" * 70)
    
    required_packages = {
        'numpy': 'numpy>=1.24.0',
        'sentence_transformers': 'sentence-transformers>=2.2.0',  # Install first (includes torch)
        'transformers': 'transformers>=4.30.0',
        'sklearn': 'scikit-learn>=1.3.0',
    }
    
    missing_packages = []
    
    for package_name, package_spec in required_packages.items():
        if check_package(package_name):
            print(f"✓ {package_name} installed")
        else:
            print(f"✗ {package_name} NOT installed")
            missing_packages.append(package_spec)
    
    print()
    
    # Install missing packages
    if missing_packages:
        print("Step 3: Installing missing packages")
        print("-" * 70)
        
        for package in missing_packages:
            if not install_package(package):
                logger.error(f"Failed to install {package}")
                return False
            print(f"✓ {package} installed")
        
        print()
    else:
        print("Step 3: All packages already installed")
        print("-" * 70)
        print("✓ No installation needed\n")
    
    # Test Neural Inference Engine
    print("Step 4: Testing Neural Inference Engine")
    print("-" * 70)
    
    try:
        from neuron_system.ai.neural_inference import NeuralInferenceEngine
        
        engine = NeuralInferenceEngine()
        print("✓ Neural Inference Engine imported successfully")
        
        # Try loading a small model
        print("\nTesting pretrained model loading...")
        print("(This may take a minute on first run)")
        
        try:
            engine.initialize_from_pretrained("distilbert-base-uncased")
            print("✓ Successfully loaded DistilBERT model")
            
            # Show model info
            info = engine.get_model_info()
            print(f"\nModel Information:")
            print(f"  Embedding Dim: {info['embedding_dim']}")
            print(f"  Attention Heads: {info['num_attention_heads']}")
            print(f"  Hidden Dim: {info['hidden_dim']}")
            print(f"  Initialized: {info['initialized']}")
            
        except Exception as e:
            logger.warning(f"Could not load pretrained model: {e}")
            logger.warning("This is OK - model will be downloaded on first use")
        
        print()
        
    except Exception as e:
        logger.error(f"Failed to test Neural Inference Engine: {e}")
        return False
    
    # Success
    print("=" * 70)
    print("SETUP COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run demo: python examples/neural_inference_demo.py")
    print("  2. Read docs: docs/NEURAL_INFERENCE.md")
    print("  3. Integrate into your system")
    print("\n" + "=" * 70 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
