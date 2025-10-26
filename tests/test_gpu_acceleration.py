"""
Test GPU Acceleration for massive dataset training.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np


def test_gpu_detection():
    """Test GPU detection and device selection."""
    print("=" * 70)
    print("TEST 1: GPU DETECTION")
    print("=" * 70)
    print()
    
    from neuron_system.engines.gpu_accelerator import GPUAccelerator
    
    # Test with GPU enabled
    gpu = GPUAccelerator(force_cpu=False)
    info = gpu.get_device_info()
    
    print(f"Device detected: {info['device_name']}")
    print(f"Device type: {info['device_type']}")
    print(f"Is GPU: {info['is_gpu']}")
    print(f"PyTorch version: {info['pytorch_version']}")
    
    if info['is_gpu']:
        print(f"\n[OK] GPU is available and will be used!")
        if 'gpu_memory_total_gb' in info:
            print(f"GPU Memory: {info['gpu_memory_total_gb']:.1f} GB")
            print(f"GPU Name: {info['gpu_name']}")
    else:
        print(f"\n[WARNING] No GPU detected, using CPU")
        print("For massive datasets, GPU is highly recommended!")
    
    print()
    
    # Test batch size optimization
    print("Batch size optimization:")
    for default_size in [32, 64, 128]:
        optimized = gpu.optimize_batch_size(default_size)
        print(f"  {default_size} → {optimized}")
    
    print()
    return gpu.is_gpu_available()


def test_compression_speed():
    """Test compression speed with and without GPU."""
    print("=" * 70)
    print("TEST 2: COMPRESSION SPEED COMPARISON")
    print("=" * 70)
    print()
    
    from neuron_system.engines.compression import CompressionEngine
    
    # Test data
    test_texts = [
        "This is a test sentence for compression.",
        "Machine learning is fascinating and powerful.",
        "Neural networks can learn complex patterns.",
        "GPU acceleration makes training much faster.",
        "Large language models require significant compute.",
    ] * 20  # 100 texts total
    
    print(f"Test data: {len(test_texts)} texts")
    print()
    
    # Test with GPU
    print("Testing with GPU enabled...")
    engine_gpu = CompressionEngine(use_gpu=True)
    
    start = time.time()
    vectors_gpu, metadata_gpu = engine_gpu.batch_compress(test_texts, batch_size=None)
    time_gpu = time.time() - start
    
    print(f"  Time: {time_gpu*1000:.2f}ms")
    print(f"  Vectors shape: {vectors_gpu.shape}")
    print(f"  Throughput: {len(test_texts)/time_gpu:.1f} texts/sec")
    
    # Get GPU stats
    stats_gpu = engine_gpu.get_performance_stats()
    if 'gpu_info' in stats_gpu:
        gpu_info = stats_gpu['gpu_info']
        print(f"  Device: {gpu_info['device_name']}")
    
    print()
    
    # Test with CPU
    print("Testing with CPU (forced)...")
    engine_cpu = CompressionEngine(use_gpu=False)
    
    start = time.time()
    vectors_cpu, metadata_cpu = engine_cpu.batch_compress(test_texts, batch_size=32)
    time_cpu = time.time() - start
    
    print(f"  Time: {time_cpu*1000:.2f}ms")
    print(f"  Vectors shape: {vectors_cpu.shape}")
    print(f"  Throughput: {len(test_texts)/time_cpu:.1f} texts/sec")
    print()
    
    # Compare
    if time_gpu < time_cpu:
        speedup = time_cpu / time_gpu
        print(f"[OK] GPU is {speedup:.1f}x FASTER than CPU!")
    else:
        print(f"[INFO] CPU was faster (small dataset, GPU overhead)")
    
    print()
    return time_gpu, time_cpu


def test_large_batch():
    """Test large batch processing with GPU."""
    print("=" * 70)
    print("TEST 3: LARGE BATCH PROCESSING")
    print("=" * 70)
    print()
    
    from neuron_system.engines.compression import CompressionEngine
    
    # Generate large test dataset
    print("Generating large test dataset...")
    test_texts = [
        f"This is test sentence number {i} with some random content about machine learning and AI."
        for i in range(1000)
    ]
    print(f"Dataset size: {len(test_texts)} texts")
    print()
    
    # Test with GPU
    engine = CompressionEngine(use_gpu=True)
    
    print("Processing with GPU acceleration...")
    start = time.time()
    vectors, metadata = engine.batch_compress(test_texts, batch_size=None)  # Auto-optimize
    elapsed = time.time() - start
    
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Vectors shape: {vectors.shape}")
    print(f"  Throughput: {len(test_texts)/elapsed:.1f} texts/sec")
    print(f"  Average per text: {elapsed*1000/len(test_texts):.2f}ms")
    
    # Get performance stats
    stats = engine.get_performance_stats()
    print(f"\nPerformance Statistics:")
    print(f"  Total compressions: {stats['total_compressions']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Average time: {stats['average_time_ms']:.2f}ms")
    
    if 'gpu_info' in stats:
        print(f"\nGPU Information:")
        gpu_info = stats['gpu_info']
        print(f"  Device: {gpu_info['device_name']}")
        if 'gpu_memory_allocated_gb' in gpu_info:
            print(f"  Memory used: {gpu_info['gpu_memory_allocated_gb']:.2f} GB")
    
    print()
    
    # Estimate for massive dataset
    texts_per_sec = len(test_texts) / elapsed
    million_texts_time = 1_000_000 / texts_per_sec
    
    print(f"Estimated time for 1 million texts: {million_texts_time/60:.1f} minutes")
    print(f"Estimated time for 10 million texts: {million_texts_time*10/3600:.1f} hours")
    print()
    
    return texts_per_sec


def main():
    """Run all GPU acceleration tests."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "GPU ACCELERATION TESTS" + " " * 26 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    try:
        # Test 1: GPU Detection
        has_gpu = test_gpu_detection()
        
        # Test 2: Speed Comparison
        time_gpu, time_cpu = test_compression_speed()
        
        # Test 3: Large Batch
        throughput = test_large_batch()
        
        # Summary
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print()
        
        if has_gpu:
            print("[✓] GPU acceleration is ENABLED and working!")
            print(f"[✓] Throughput: {throughput:.1f} texts/second")
            print()
            print("Your system is ready for massive Reddit dataset training!")
            print("Recommended command:")
            print("  python cli.py reddit dataset.json --batch-size 1000")
        else:
            print("[!] GPU acceleration is NOT available")
            print(f"[!] CPU throughput: {throughput:.1f} texts/second")
            print()
            print("For massive datasets, GPU is highly recommended!")
            print("Consider:")
            print("  - Installing CUDA for NVIDIA GPUs")
            print("  - Using a cloud GPU instance (AWS, GCP, Azure)")
            print("  - Processing smaller batches with --max-conversations")
        
        print()
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
