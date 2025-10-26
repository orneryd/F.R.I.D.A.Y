"""
Test if sentence-transformers model loads correctly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron_system.engines.compression import CompressionEngine


def test_model_loading():
    print("=" * 70)
    print("TESTING MODEL LOADING")
    print("=" * 70)
    print()
    
    try:
        print("1. Creating CompressionEngine...")
        engine = CompressionEngine()
        print("   ✓ Engine created")
        print()
        
        print("2. Testing compression (this will load the model)...")
        test_text = "Hello, this is a test."
        vector, metadata = engine.compress(test_text)
        
        print(f"   ✓ Compression successful")
        print(f"   Vector shape: {vector.shape}")
        print(f"   Vector dim: {metadata['vector_dim']}")
        print(f"   Time: {metadata['elapsed_time_ms']:.2f}ms")
        print()
        
        print("3. Testing multiple compressions...")
        texts = [
            "What are you?",
            "Who are you?",
            "Tell me about yourself"
        ]
        
        for i, text in enumerate(texts, 1):
            vec, meta = engine.compress(text)
            print(f"   {i}. '{text}' -> {vec.shape} ({meta['elapsed_time_ms']:.2f}ms)")
        
        print()
        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model_loading()
