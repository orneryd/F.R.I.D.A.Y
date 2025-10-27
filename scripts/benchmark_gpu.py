"""
GPU vs CPU Performance Benchmark
"""

import time
import torch
from neuron_system.engines.compression import CompressionEngine

def benchmark_compression(use_gpu: bool, num_texts: int = 100, num_runs: int = 3):
    """Benchmark compression speed."""
    print(f"\n{'='*70}")
    print(f"Testing: {'GPU (CUDA)' if use_gpu else 'CPU'}")
    print(f"{'='*70}")
    
    # Initialize engine
    engine = CompressionEngine(use_gpu=use_gpu)
    
    # Test data
    test_texts = [
        f"This is test sentence number {i} for performance testing. "
        f"It contains some meaningful content to compress properly."
        for i in range(num_texts)
    ]
    
    # Warmup (wichtig für GPU!)
    print("Warming up...")
    engine.batch_compress(test_texts[:10], batch_size=None)
    
    if use_gpu:
        torch.cuda.synchronize()
    
    # Multiple runs for stability
    times = []
    print(f"Running {num_runs} benchmarks with {num_texts} texts each...")
    
    for run in range(num_runs):
        start = time.time()
        vectors, metadata = engine.batch_compress(test_texts, batch_size=None)
        
        if use_gpu:
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.2f}s ({num_texts/elapsed:.1f} texts/sec)")
    
    # Use best time (common practice for benchmarks)
    elapsed = min(times)
    avg_elapsed = sum(times) / len(times)
    
    # Results
    throughput = num_texts / elapsed
    avg_throughput = num_texts / avg_elapsed
    
    print(f"\nResults (best of {num_runs} runs):")
    print(f"  Texts processed:    {num_texts}")
    print(f"  Best time:          {elapsed:.2f}s")
    print(f"  Average time:       {avg_elapsed:.2f}s")
    print(f"  Best throughput:    {throughput:.1f} texts/sec")
    print(f"  Avg throughput:     {avg_throughput:.1f} texts/sec")
    print(f"  Avg per text:       {elapsed*1000/num_texts:.1f}ms")
    
    # Memory stats for GPU
    if use_gpu and torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated:          {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Cached:             {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    return throughput, elapsed


def main():
    print("\n" + "="*70)
    print("GPU vs CPU PERFORMANCE BENCHMARK")
    print("="*70)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠ No GPU detected - will only test CPU")
    
    num_texts = 2000  # Mehr Texte für besseren Benchmark
    num_runs = 5  # Mehrere Durchläufe für Stabilität
    
    # Test CPU
    cpu_throughput, cpu_time = benchmark_compression(use_gpu=False, num_texts=num_texts, num_runs=num_runs)
    
    # Test GPU (if available)
    if torch.cuda.is_available():
        gpu_throughput, gpu_time = benchmark_compression(use_gpu=True, num_texts=num_texts, num_runs=num_runs)
        
        # Comparison
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"\nSpeedup: {gpu_throughput / cpu_throughput:.2f}x faster with GPU")
        print(f"Time saved: {cpu_time - gpu_time:.2f}s ({(1 - gpu_time/cpu_time)*100:.1f}% faster)")
        
        # Estimates for large datasets
        print(f"\nEstimated time for 100K conversations:")
        cpu_hours = 100_000 / cpu_throughput / 3600
        gpu_hours = 100_000 / gpu_throughput / 3600
        saved_hours = cpu_hours - gpu_hours
        
        if cpu_hours < 1:
            print(f"  CPU:  {100_000 / cpu_throughput / 60:.1f} minutes")
            print(f"  GPU:  {100_000 / gpu_throughput / 60:.1f} minutes")
            print(f"  Saved: {saved_hours * 60:.1f} minutes")
        else:
            print(f"  CPU:  {cpu_hours:.1f} hours")
            print(f"  GPU:  {gpu_hours:.1f} hours")
            print(f"  Saved: {saved_hours:.1f} hours")
    
    print()


if __name__ == "__main__":
    main()
