"""
Example: GPU-Accelerated Training on Large Datasets

This example demonstrates how to use GPU acceleration for training
on massive Reddit conversation datasets.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron_system.config.settings import Settings
from main import ApplicationContainer
from neuron_system.ai.language_model import LanguageModel
from neuron_system.ai.smart_trainer import SmartTrainer
from neuron_system.ai.reddit_loader import RedditLoader, generate_test_data
from neuron_system.engines.compression import CompressionEngine
from neuron_system.engines.gpu_accelerator import GPUAccelerator
import time


def main():
    """Example of GPU-accelerated training."""
    print("=" * 70)
    print("GPU-ACCELERATED TRAINING EXAMPLE")
    print("=" * 70)
    print()
    
    # Step 1: Check GPU availability
    print("Step 1: Checking GPU availability...")
    gpu = GPUAccelerator()
    gpu_info = gpu.get_device_info()
    
    print(f"  Device: {gpu_info['device_name']}")
    print(f"  Type: {gpu_info['device_type']}")
    print(f"  Is GPU: {gpu_info['is_gpu']}")
    
    if gpu_info['is_gpu']:
        print(f"  ✓ GPU acceleration is ENABLED")
        if 'gpu_memory_total_gb' in gpu_info:
            print(f"  Memory: {gpu_info['gpu_memory_total_gb']:.1f} GB")
    else:
        print(f"  ⚠ GPU not available, using CPU")
    print()
    
    # Step 2: Generate test dataset
    print("Step 2: Generating test dataset...")
    test_file = "gpu_test_dataset.json"
    num_samples = 5000  # 5K conversations for demo
    
    generate_test_data(test_file, num_samples=num_samples)
    print(f"  ✓ Generated {num_samples} test conversations")
    print()
    
    # Step 3: Initialize system with GPU
    print("Step 3: Initializing system with GPU support...")
    settings = Settings(
        database_path="gpu_test.db",
        spatial_bounds_min=(-500.0, -500.0, -500.0),
        spatial_bounds_max=(500.0, 500.0, 500.0)
    )
    
    container = ApplicationContainer(settings)
    container.initialize()
    
    # Create compression engine with GPU
    compression_engine = CompressionEngine(use_gpu=True)
    
    # Create language model
    language_model = LanguageModel(
        container.graph,
        compression_engine,
        container.query_engine,
        container.training_engine
    )
    
    print(f"  ✓ System initialized")
    print()
    
    # Step 4: Train with GPU acceleration
    print("Step 4: Training with GPU acceleration...")
    print("-" * 70)
    
    smart_trainer = SmartTrainer(language_model)
    loader = RedditLoader()
    
    # Load conversations
    conversations = loader.load_from_file(test_file, max_conversations=num_samples)
    print(f"Loaded {len(conversations)} conversations")
    print()
    
    # Train with timing
    start_time = time.time()
    
    for i, (question, answer) in enumerate(conversations, 1):
        success, reason = smart_trainer.train_conversation(question, answer)
        
        # Progress every 500
        if i % 500 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (len(conversations) - i) / rate if rate > 0 else 0
            
            stats = smart_trainer.get_statistics()
            print(f"  Progress: {i}/{len(conversations)} "
                  f"({i/len(conversations)*100:.1f}%) "
                  f"- {rate:.1f} conv/sec "
                  f"- ETA: {eta:.1f}s")
            print(f"    Added: {stats['successfully_added']}, "
                  f"Duplicates: {stats['duplicates_found']}, "
                  f"Rejected: {stats['total_rejected']}")
    
    total_time = time.time() - start_time
    
    print()
    print("-" * 70)
    print()
    
    # Step 5: Results
    print("Step 5: Training Results")
    print("=" * 70)
    
    stats = smart_trainer.get_statistics()
    
    print(f"Training Statistics:")
    print(f"  Total processed:    {stats['total_processed']:,}")
    print(f"  Successfully added: {stats['successfully_added']:,}")
    print(f"  Duplicates found:   {stats['duplicates_found']:,}")
    print(f"  Quality rejected:   {stats['quality_rejected']:,}")
    print(f"  Logic rejected:     {stats['logic_rejected']:,}")
    print(f"  Success rate:       {stats['success_rate']:.1%}")
    print()
    
    print(f"Performance:")
    print(f"  Total time:         {total_time:.2f}s")
    print(f"  Throughput:         {len(conversations)/total_time:.1f} conv/sec")
    print(f"  Average per conv:   {total_time*1000/len(conversations):.2f}ms")
    print()
    
    # Get compression engine stats
    comp_stats = compression_engine.get_performance_stats()
    print(f"Compression Engine:")
    print(f"  Total compressions: {comp_stats['total_compressions']:,}")
    print(f"  Average time:       {comp_stats['average_time_ms']:.2f}ms")
    print(f"  Success rate:       {comp_stats['success_rate']:.1%}")
    
    if 'gpu_info' in comp_stats:
        print(f"\nGPU Information:")
        gpu_info = comp_stats['gpu_info']
        print(f"  Device:             {gpu_info['device_name']}")
        if 'gpu_memory_allocated_gb' in gpu_info:
            print(f"  Memory used:        {gpu_info['gpu_memory_allocated_gb']:.2f} GB")
    
    print()
    
    # Step 6: Estimate for larger datasets
    print("Step 6: Estimates for Larger Datasets")
    print("=" * 70)
    
    rate = len(conversations) / total_time
    
    estimates = [
        (100_000, "100K"),
        (1_000_000, "1M"),
        (10_000_000, "10M"),
    ]
    
    for size, label in estimates:
        time_seconds = size / rate
        time_minutes = time_seconds / 60
        time_hours = time_minutes / 60
        
        if time_hours >= 1:
            time_str = f"{time_hours:.1f} hours"
        elif time_minutes >= 1:
            time_str = f"{time_minutes:.1f} minutes"
        else:
            time_str = f"{time_seconds:.1f} seconds"
        
        print(f"  {label:>6} conversations: {time_str}")
    
    print()
    
    # Cleanup
    container.shutdown()
    
    # Remove test files
    if os.path.exists(test_file):
        os.remove(test_file)
    if os.path.exists("gpu_test.db"):
        os.remove("gpu_test.db")
    
    print("=" * 70)
    print("EXAMPLE COMPLETE!")
    print("=" * 70)
    print()
    
    if gpu_info['is_gpu']:
        print("✓ Your system is ready for massive dataset training!")
        print()
        print("Next steps:")
        print("  1. Get a large Reddit dataset")
        print("  2. Run: python cli.py reddit dataset.json --batch-size 256")
        print("  3. Watch it train at lightning speed! ⚡")
    else:
        print("⚠ For massive datasets, GPU acceleration is recommended!")
        print()
        print("Options:")
        print("  1. Install CUDA for NVIDIA GPUs")
        print("  2. Use a cloud GPU instance")
        print("  3. Process smaller batches with --max-conversations")
    
    print()


if __name__ == "__main__":
    main()
