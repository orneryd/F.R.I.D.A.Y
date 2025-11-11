"""
Setup Script for Friday's Local Model.

Downloads Qwen3-0.6B model for local text generation.
"""

import logging
from neuron_system.ai.local_model import LocalModelManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main():
    """Download and setup the model."""
    print_header("FRIDAY LOCAL MODEL SETUP")
    
    print("\n📦 Model: Qwen3-0.6B")
    print("📏 Size: ~600MB")
    print("🚀 Speed: Very fast (0.6B parameters)")
    print("💾 Location: models/Qwen_Qwen3-0.6B/")
    
    print("\n⚠️  This will download ~600MB of data.")
    response = input("\nContinue? (y/n): ")
    
    if response.lower() != 'y':
        print("Setup cancelled.")
        return
    
    print_header("DOWNLOADING MODEL")
    
    manager = LocalModelManager(model_dir="models")
    
    success = manager.download_model(
        model_name="Qwen/Qwen3-0.6B",
        force_download=False
    )
    
    if success:
        print_header("SETUP COMPLETE")
        
        info = manager.get_model_info()
        
        print("\n✅ Model downloaded successfully!")
        print(f"\n📊 Model Information:")
        print(f"   Name: {info['name']}")
        print(f"   Device: {info['device']}")
        print(f"   Parameters: {info['parameters']:,}")
        print(f"   Location: {info['location']}")
        
        print("\n🧪 Testing model...")
        test_response = manager.generate_response(
            "Hello! Who are you?",
            max_length=50
        )
        
        if test_response:
            print(f"\n✓ Test successful!")
            print(f"   Response: {test_response[:100]}...")
        
        print("\n🚀 Ready to use!")
        print("\nFriday will now use this local model for text generation.")
        print("No external services (Ollama) needed!")
        
        print("\n💡 Usage:")
        print("   python cli.py chat")
        
    else:
        print_header("SETUP FAILED")
        print("\n❌ Failed to download model.")
        print("\n🔧 Troubleshooting:")
        print("   1. Check internet connection")
        print("   2. Install dependencies: pip install transformers torch")
        print("   3. Try again: python setup_model.py")


if __name__ == "__main__":
    main()
