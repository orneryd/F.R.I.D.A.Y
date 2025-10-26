"""
F.R.I.D.A.Y AI - Command Line Interface

Easy-to-use CLI for training, chatting, and managing the AI.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ApplicationContainer
from neuron_system.config.settings import Settings
from neuron_system.ai.language_model import LanguageModel
from neuron_system.ai.incremental_trainer import IncrementalTrainer
from neuron_system.ai.dataset_loader import DatasetLoader
from neuron_system.ai.pretraining import PreTrainingLoader


def cmd_train_full(args):
    """Full training from scratch."""
    print("=" * 70)
    print("FULL AI TRAINING")
    print("=" * 70)
    print()
    
    if os.path.exists(args.database):
        print(f"WARNING: Database '{args.database}' already exists!")
        response = input("Overwrite? (yes/no): ")
        if response.lower() != "yes":
            print("Cancelled.")
            return
        os.remove(args.database)
    
    print("Starting full training...")
    print()
    
    settings = Settings(
        database_path=args.database,
        spatial_bounds_min=(-500.0, -500.0, -500.0),
        spatial_bounds_max=(500.0, 500.0, 500.0),
        enable_auto_save=True,
        auto_save_interval_seconds=300
    )
    
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        language_model = LanguageModel(
            container.graph,
            container.compression_engine,
            container.query_engine,
            container.training_engine
        )
        
        # Load built-in knowledge
        print("1. Loading built-in knowledge...")
        pre_loader = PreTrainingLoader(language_model, spatial_distribution="clustered")
        built_in_count = pre_loader.load_english_knowledge(create_connections=False, batch_size=20)
        print(f"   [OK] Loaded {built_in_count} items")
        
        # Load conversation knowledge
        print("2. Loading conversation knowledge...")
        from neuron_system.ai.conversation_knowledge import ConversationKnowledge
        from neuron_system.ai.natural_dialogue import NaturalDialogue
        from neuron_system.ai.additional_knowledge import AdditionalKnowledge
        from neuron_system.ai.reasoning_knowledge import ReasoningKnowledge
        
        for knowledge_class in [ConversationKnowledge, NaturalDialogue, AdditionalKnowledge, ReasoningKnowledge]:
            items = knowledge_class.get_all_knowledge()
            for item in items:
                language_model.learn(text=item['text'], tags=item['tags'], create_connections=False)
        print(f"   [OK] Loaded conversation knowledge")
        
        # Load datasets if requested
        if args.with_datasets:
            print("3. Loading external datasets...")
            loader = DatasetLoader(language_model)
            loader.load_from_huggingface(
                dataset_name="wikitext",
                split="train",
                text_field="text",
                max_samples=args.max_samples,
                batch_size=100
            )
            print(f"   [OK] Loaded datasets")
        
        # Create connections
        print("4. Creating connections...")
        stats = language_model.get_statistics()
        total = stats['total_neurons']
        
        loader = DatasetLoader(language_model)
        if total > 2000:
            loader.create_connections_batch(top_k=3, save_interval=500, max_neurons=2000)
        else:
            loader.create_connections_batch(top_k=3, save_interval=200)
        print("   [OK] Connections created")
        
        # Save
        print("5. Saving to database...")
        loader.save_to_database()
        print("   [OK] Saved")
        
        # Statistics
        final_stats = language_model.get_statistics()
        print()
        print("=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Neurons: {final_stats['total_neurons']}")
        print(f"Synapses: {final_stats['total_synapses']}")
        print(f"Connectivity: {final_stats['average_connectivity']:.2f}")
        print(f"Database: {args.database}")
        print()
        
    finally:
        container.shutdown()


def cmd_train_incremental(args):
    """Incremental training - update existing AI."""
    print("=" * 70)
    print("INCREMENTAL TRAINING")
    print("=" * 70)
    print()
    
    if not os.path.exists(args.database):
        print(f"ERROR: Database '{args.database}' not found!")
        print("Run 'python cli.py train' first to create initial AI.")
        return
    
    settings = Settings(
        database_path=args.database,
        spatial_bounds_min=(-500.0, -500.0, -500.0),
        spatial_bounds_max=(500.0, 500.0, 500.0),
        enable_auto_save=True
    )
    
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        language_model = LanguageModel(
            container.graph,
            container.compression_engine,
            container.query_engine,
            container.training_engine
        )
        
        print("1. Loading existing AI...")
        stats = language_model.get_statistics()
        print(f"   Loaded: {stats['total_neurons']} neurons, {stats['total_synapses']} synapses")
        
        print("2. Initializing incremental trainer...")
        trainer = IncrementalTrainer(language_model)
        print(f"   Indexed: {len(trainer.existing_texts)} items")
        
        print("3. Loading new knowledge...")
        from neuron_system.ai.conversation_knowledge import ConversationKnowledge
        from neuron_system.ai.natural_dialogue import NaturalDialogue
        from neuron_system.ai.reasoning_knowledge import ReasoningKnowledge
        from neuron_system.ai.response_examples import ResponseExamples
        
        all_items = (
            ConversationKnowledge.get_all_knowledge() + 
            NaturalDialogue.get_all_knowledge() +
            ReasoningKnowledge.get_all_knowledge() +
            ResponseExamples.get_all_examples()
        )
        
        print("4. Updating knowledge base...")
        update_stats = trainer.batch_add_or_update(all_items, batch_size=50)
        print(f"   Added: {update_stats['added']}")
        print(f"   Updated: {update_stats['updated']}")
        print(f"   Skipped: {update_stats['skipped']}")
        
        print("5. Removing duplicates...")
        removed = trainer.remove_duplicate_knowledge()
        print(f"   Removed: {removed} duplicates")
        
        print("6. Saving...")
        trainer.save_to_database()
        print("   [OK] Saved")
        
        final_stats = trainer.get_statistics()
        print()
        print("=" * 70)
        print("UPDATE COMPLETE!")
        print("=" * 70)
        print(f"Neurons: {final_stats['total_neurons']}")
        print(f"Synapses: {final_stats['total_synapses']}")
        print(f"Connectivity: {final_stats['average_connectivity']:.2f}")
        print()
        
    finally:
        container.shutdown()


def cmd_chat(args):
    """Interactive chat with the AI (with memory support)."""
    if not os.path.exists(args.database):
        print(f"ERROR: Database '{args.database}' not found!")
        print("Run 'python cli.py train' first.")
        return
    
    settings = Settings(
        database_path=args.database,
        spatial_bounds_min=(-500.0, -500.0, -500.0),
        spatial_bounds_max=(500.0, 500.0, 500.0)
    )
    
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        from neuron_system.neuron_types.memory_neuron import MemoryManager
        
        language_model = LanguageModel(
            container.graph,
            container.compression_engine,
            container.query_engine,
            container.training_engine
        )
        
        # Initialize memory manager
        memory_manager = MemoryManager(
            container.graph,
            container.compression_engine
        )
        
        print("=" * 70)
        print("F.R.I.D.A.Y AI - Interactive Chat (with Memory)")
        print("=" * 70)
        print()
        print("Type 'exit' or 'quit' to end the conversation")
        print("Type 'stats' to see AI statistics")
        print("Type 'memory' to see conversation memory")
        print()
        
        conversation_count = 0
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("AI: Goodbye! Have a great day!")
                    break
                
                if user_input.lower() == 'stats':
                    stats = language_model.get_statistics()
                    print(f"\nAI Statistics:")
                    print(f"  Neurons: {stats['total_neurons']:,}")
                    print(f"  Synapses: {stats['total_synapses']:,}")
                    print(f"  Connectivity: {stats['average_connectivity']:.2f}")
                    print()
                    continue
                
                if user_input.lower() == 'memory':
                    mem_stats = memory_manager.get_statistics()
                    print(f"\nMemory Statistics:")
                    print(f"  Total memories: {mem_stats['total_memories']}")
                    print(f"  By type: {mem_stats['by_type']}")
                    print(f"  Avg importance: {mem_stats['average_importance']:.2f}")
                    print(f"  Total accesses: {mem_stats['total_accesses']}")
                    print()
                    continue
                
                # Retrieve relevant memories for context
                relevant_memories = memory_manager.retrieve_memories(
                    user_input,
                    top_k=3,
                    min_importance=0.3
                )
                
                # Build context from memories
                memory_context = ""
                if relevant_memories:
                    memory_context = "\n[Context from previous conversation: "
                    memory_context += "; ".join([
                        m.source_data[:50] + "..." if len(m.source_data) > 50 else m.source_data
                        for m in relevant_memories
                    ])
                    memory_context += "]"
                
                # Generate response
                response = language_model.generate_response(
                    user_input + memory_context,
                    context_size=args.context_size,
                    min_activation=args.min_activation
                )
                
                print(f"AI: {response}")
                print()
                
                # Store conversation in memory
                conversation_count += 1
                
                # Store user message
                memory_manager.create_memory(
                    content=f"User said: {user_input}",
                    memory_type="short-term",
                    context={'turn': conversation_count, 'speaker': 'user'},
                    importance=0.5
                )
                
                # Store AI response
                memory_manager.create_memory(
                    content=f"AI responded: {response}",
                    memory_type="short-term",
                    context={'turn': conversation_count, 'speaker': 'ai'},
                    importance=0.5
                )
                
                # Consolidate memories every 10 turns
                if conversation_count % 10 == 0:
                    memory_manager.consolidate_memories()
                
            except KeyboardInterrupt:
                print("\n\nAI: Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
    finally:
        container.shutdown()


def cmd_test(args):
    """Test the AI with predefined questions."""
    if not os.path.exists(args.database):
        print(f"ERROR: Database '{args.database}' not found!")
        return
    
    settings = Settings(
        database_path=args.database,
        spatial_bounds_min=(-500.0, -500.0, -500.0),
        spatial_bounds_max=(500.0, 500.0, 500.0)
    )
    
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        language_model = LanguageModel(
            container.graph,
            container.compression_engine,
            container.query_engine,
            container.training_engine
        )
        
        print("=" * 70)
        print("AI TEST")
        print("=" * 70)
        print()
        
        test_questions = [
            "What are you?",
            "Who are you?",
            "Can you help me?",
            "How are you?",
            "What is AI?",
            "Do you have feelings?",
            "Can you learn?",
            "Are you always right?",
            "Thank you",
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"{i}. Q: {question}")
            response = language_model.generate_response(
                question,
                context_size=5,
                min_activation=0.1
            )
            print(f"   A: {response}")
            print()
        
        stats = language_model.get_statistics()
        print("=" * 70)
        print("STATISTICS")
        print("=" * 70)
        print(f"Neurons: {stats['total_neurons']}")
        print(f"Synapses: {stats['total_synapses']}")
        print(f"Connectivity: {stats['average_connectivity']:.2f}")
        print()
        
    finally:
        container.shutdown()


def cmd_stats(args):
    """Show AI statistics."""
    if not os.path.exists(args.database):
        print(f"ERROR: Database '{args.database}' not found!")
        return
    
    settings = Settings(
        database_path=args.database,
        spatial_bounds_min=(-500.0, -500.0, -500.0),
        spatial_bounds_max=(500.0, 500.0, 500.0)
    )
    
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        language_model = LanguageModel(
            container.graph,
            container.compression_engine,
            container.query_engine,
            container.training_engine
        )
        
        stats = language_model.get_statistics()
        
        print("=" * 70)
        print("AI STATISTICS")
        print("=" * 70)
        print(f"Database: {args.database}")
        print()
        
        print("Neurons:")
        print(f"  Total:      {stats['total_neurons']:,}")
        print(f"  Knowledge:  {stats['knowledge_neurons']:,}")
        print(f"  Reasoning:  {stats['reasoning_neurons']:,}")
        print(f"  Memory:     {stats['memory_neurons']:,}")
        print(f"  Tool:       {stats['tool_neurons']:,}")
        if stats['other_neurons'] > 0:
            print(f"  Other:      {stats['other_neurons']:,}")
        print()
        
        print("Network:")
        print(f"  Synapses:   {stats['total_synapses']:,}")
        print(f"  Connectivity: {stats['average_connectivity']:.2f} synapses/neuron")
        print(f"  Avg Importance: {stats['average_importance']:.3f}")
        print()
        
        if stats['tool_neurons'] > 0:
            print("Tools:")
            print(f"  Total executions: {stats['tool_execution_count']:,}")
            print()
        
        # File size
        if os.path.exists(args.database):
            size_mb = os.path.getsize(args.database) / (1024 * 1024)
            print(f"Database Size: {size_mb:.2f} MB")
        
        print()
        
    finally:
        container.shutdown()


def cmd_train_reddit(args):
    """Train from Reddit dataset with smart deduplication."""
    print("=" * 70)
    print("REDDIT DATASET TRAINING")
    print("=" * 70)
    print()
    
    if not os.path.exists(args.database):
        print(f"ERROR: Database '{args.database}' not found!")
        print("Run 'python cli.py train' first to create the database.")
        return
    
    if not os.path.exists(args.file):
        print(f"ERROR: Reddit data file '{args.file}' not found!")
        return
    
    settings = Settings(
        database_path=args.database,
        spatial_bounds_min=(-500.0, -500.0, -500.0),
        spatial_bounds_max=(500.0, 500.0, 500.0)
    )
    
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        from neuron_system.ai.smart_trainer import SmartTrainer
        from neuron_system.ai.reddit_loader import RedditLoader
        from neuron_system.engines.compression import CompressionEngine
        import json
        
        # Initialize compression engine with GPU support
        use_gpu = not args.no_gpu
        compression_engine = CompressionEngine(use_gpu=use_gpu)
        
        # Show GPU info
        if use_gpu:
            from neuron_system.engines.gpu_accelerator import GPUAccelerator
            gpu = GPUAccelerator()
            gpu_info = gpu.get_device_info()
            print(f"GPU Acceleration: {gpu_info['device_name']}")
            if gpu_info['is_gpu']:
                print(f"  Device: {gpu_info['device_type'].upper()}")
                if 'gpu_memory_total_gb' in gpu_info:
                    print(f"  Memory: {gpu_info['gpu_memory_total_gb']:.1f} GB")
                # Optimize batch size
                optimized_batch = gpu.optimize_batch_size(args.batch_size)
                if optimized_batch != args.batch_size:
                    print(f"  Batch size optimized: {args.batch_size} → {optimized_batch}")
                    args.batch_size = optimized_batch
            print()
        else:
            print("GPU Acceleration: DISABLED (using CPU)")
            print()
        
        language_model = LanguageModel(
            container.graph,
            compression_engine,
            container.query_engine,
            container.training_engine
        )
        
        smart_trainer = SmartTrainer(language_model)
        smart_trainer.min_quality_score = args.quality_threshold
        smart_trainer.similarity_threshold = args.similarity_threshold
        
        loader = RedditLoader()
        
        # Get initial state
        initial_neurons = len(container.graph.neurons)
        print(f"Initial neurons: {initial_neurons:,}")
        print(f"Quality threshold: {args.quality_threshold}")
        print(f"Similarity threshold: {args.similarity_threshold}")
        print()
        
        # Load conversations
        print(f"Loading from: {args.file}")
        print(f"Max conversations: {args.max_conversations if args.max_conversations else 'unlimited'}")
        print(f"Min score: {args.min_score}")
        print()
        
        conversations = loader.load_from_file(
            args.file,
            max_conversations=args.max_conversations,
            min_score=args.min_score
        )
        
        print(f"Loaded {len(conversations)} conversations")
        print()
        
        if not conversations:
            print("No conversations loaded. Check file format.")
            return
        
        # Train with progress tracking and checkpoints
        print("Training with smart deduplication...")
        print("-" * 70)
        
        checkpoint_interval = args.checkpoint
        last_checkpoint = 0
        
        for i, (question, answer) in enumerate(conversations, 1):
            success, reason = smart_trainer.train_conversation(question, answer)
            
            # Show progress
            if args.verbose:
                status = "[OK]" if success else "[SKIP]"
                print(f"{i}. {status} Q: {question[:50]}...")
                if not success and args.verbose:
                    print(f"   Reason: {reason}")
            elif i % 100 == 0:
                stats = smart_trainer.get_statistics()
                print(f"  Progress: {i}/{len(conversations)} "
                      f"(Added: {stats['successfully_added']}, "
                      f"Duplicates: {stats['duplicates_found']}, "
                      f"Rejected: {stats['total_rejected']})")
            
            # Checkpoint
            if i - last_checkpoint >= checkpoint_interval:
                print(f"\n  [CHECKPOINT] Saving at {i}/{len(conversations)}...")
                if hasattr(container.graph, 'neuron_store'):
                    # Save new neurons
                    new_neurons = [
                        n for n in container.graph.neurons.values()
                        if n.id not in container.graph.neuron_store._cache
                    ]
                    if new_neurons:
                        for neuron in new_neurons:
                            try:
                                container.graph.neuron_store.create(neuron)
                            except:
                                pass  # Already exists
                print(f"  [OK] Checkpoint saved\n")
                last_checkpoint = i
        
        # Final statistics
        final_neurons = len(container.graph.neurons)
        stats = smart_trainer.get_statistics()
        
        print()
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print()
        print(f"Conversations processed: {stats['total_processed']:,}")
        print(f"Successfully added:      {stats['successfully_added']:,}")
        print(f"Duplicates filtered:     {stats['duplicates_found']:,}")
        print(f"Quality rejected:        {stats['quality_rejected']:,}")
        print(f"Logic rejected:          {stats['logic_rejected']:,}")
        print(f"Success rate:            {stats['success_rate']:.1%}")
        print()
        print(f"Neurons before:          {initial_neurons:,}")
        print(f"Neurons after:           {final_neurons:,}")
        print(f"Neurons added:           {final_neurons - initial_neurons:,}")
        print()
        
        # Save final state
        if args.save:
            print("Saving final state to database...")
            # Save all new neurons
            if hasattr(container.graph, 'neuron_store'):
                new_neurons = [
                    n for n in container.graph.neurons.values()
                    if n.id not in container.graph.neuron_store._cache
                ]
                saved = 0
                for neuron in new_neurons:
                    try:
                        container.graph.neuron_store.create(neuron)
                        saved += 1
                    except:
                        pass
                print(f"[OK] Saved {saved} new neurons")
            print()
        else:
            print("Changes NOT saved (use --save to persist)")
            print()
        
    finally:
        container.shutdown()


def cmd_gpu_info(args):
    """Show GPU information and capabilities."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 24 + "GPU INFORMATION" + " " * 29 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    try:
        from neuron_system.engines.gpu_accelerator import GPUAccelerator
        from neuron_system.engines.compression import CompressionEngine
        import torch
        import subprocess
        
        # Check for CUDA Toolkit installation
        cuda_toolkit_version = None
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                # Parse CUDA version from nvcc output
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        # Extract version like "12.9" from "release 12.9, V12.9.41"
                        parts = line.split('release')
                        if len(parts) > 1:
                            version_part = parts[1].split(',')[0].strip()
                            cuda_toolkit_version = version_part
                        break
        except FileNotFoundError:
            pass  # nvcc not in PATH
        except Exception:
            pass  # Other error
        
        # Initialize GPU accelerator
        gpu = GPUAccelerator()
        info = gpu.get_device_info()
        
        # Basic Info
        print("=" * 70)
        print("DEVICE INFORMATION")
        print("=" * 70)
        print()
        
        print(f"Device Name:        {info['device_name']}")
        print(f"Device Type:        {info['device_type'].upper()}")
        print(f"PyTorch Version:    {info['pytorch_version']}")
        print(f"GPU Available:      {'✓ YES' if info['is_gpu'] else '✗ NO'}")
        print()
        
        # GPU-specific info
        if info['is_gpu']:
            if info['device_type'] == 'cuda':
                print("=" * 70)
                print("NVIDIA CUDA INFORMATION")
                print("=" * 70)
                print()
                
                # PyTorch CUDA version
                pytorch_cuda = info.get('cuda_version', 'N/A')
                print(f"PyTorch CUDA:       {pytorch_cuda}")
                
                # CUDA Toolkit version (from nvcc)
                if cuda_toolkit_version:
                    print(f"CUDA Toolkit:       {cuda_toolkit_version} (nvcc detected)")
                    
                    # Check version compatibility
                    if pytorch_cuda and pytorch_cuda != 'N/A':
                        pytorch_major = pytorch_cuda.split('.')[0]
                        toolkit_major = cuda_toolkit_version.split('.')[0]
                        
                        if pytorch_major != toolkit_major:
                            print(f"                    ⚠ Version mismatch with PyTorch CUDA!")
                            print(f"                    Consider: pip install torch --index-url https://download.pytorch.org/whl/cu{toolkit_major}21")
                else:
                    print(f"CUDA Toolkit:       Not detected (nvcc not in PATH)")
                    print(f"                    ℹ PyTorch includes CUDA runtime, toolkit optional")
                
                print(f"GPU Count:          {info.get('gpu_count', 0)}")
                print(f"GPU Name:           {info.get('gpu_name', 'N/A')}")
                print(f"Total Memory:       {info.get('gpu_memory_total_gb', 0):.2f} GB")
                print(f"Allocated Memory:   {info.get('gpu_memory_allocated_gb', 0):.2f} GB")
                print(f"Cached Memory:      {info.get('gpu_memory_cached_gb', 0):.2f} GB")
                print()
                
                # Compute capability
                if torch.cuda.is_available():
                    capability = torch.cuda.get_device_capability(0)
                    print(f"Compute Capability: {capability[0]}.{capability[1]}")
                    
                    # Check if compute capability is supported
                    if capability[0] >= 3 and capability[1] >= 5:
                        print(f"Status:             ✓ Supported (>= 3.5)")
                    else:
                        print(f"Status:             ⚠ May not be supported (< 3.5)")
                    print()
            
            elif info['device_type'] == 'mps':
                print("=" * 70)
                print("APPLE SILICON MPS INFORMATION")
                print("=" * 70)
                print()
                print("Apple Silicon GPU detected (M1/M2/M3)")
                print("Using Metal Performance Shaders (MPS)")
                print()
        else:
            print("=" * 70)
            print("CPU MODE")
            print("=" * 70)
            print()
            print("⚠ No GPU detected - running in CPU mode")
            print()
            print("For GPU acceleration:")
            print("  • NVIDIA GPU: Install CUDA and PyTorch with CUDA support")
            print("  • Apple Silicon: PyTorch with MPS support (usually pre-installed)")
            print()
        
        # Batch size recommendations
        print("=" * 70)
        print("BATCH SIZE RECOMMENDATIONS")
        print("=" * 70)
        print()
        
        batch_sizes = [32, 64, 128, 256, 512]
        print("Default → Optimized")
        print("-" * 30)
        for size in batch_sizes:
            optimized = gpu.optimize_batch_size(size)
            status = "✓" if optimized > size else "→"
            print(f"  {size:>3} → {optimized:>3}  {status}")
        print()
        
        # Performance test
        if args.test:
            print("=" * 70)
            print("PERFORMANCE TEST")
            print("=" * 70)
            print()
            
            print("Testing compression speed...")
            engine = CompressionEngine(use_gpu=True)
            
            # Test data
            test_texts = [
                f"This is test sentence number {i} for GPU performance testing."
                for i in range(100)
            ]
            
            import time
            start = time.time()
            vectors, metadata = engine.batch_compress(test_texts, batch_size=None)
            elapsed = time.time() - start
            
            throughput = len(test_texts) / elapsed
            
            print(f"  Texts processed:    {len(test_texts)}")
            print(f"  Time taken:         {elapsed*1000:.2f}ms")
            print(f"  Throughput:         {throughput:.1f} texts/sec")
            print(f"  Avg per text:       {elapsed*1000/len(test_texts):.2f}ms")
            print()
            
            # Estimates
            print("Estimated training times:")
            print("-" * 30)
            for size, label in [(100_000, "100K"), (1_000_000, "1M"), (10_000_000, "10M")]:
                time_sec = size / throughput
                if time_sec >= 3600:
                    time_str = f"{time_sec/3600:.1f} hours"
                elif time_sec >= 60:
                    time_str = f"{time_sec/60:.1f} minutes"
                else:
                    time_str = f"{time_sec:.1f} seconds"
                print(f"  {label:>6} conversations: {time_str}")
            print()
        
        # Summary
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print()
        
        if info['is_gpu']:
            print("✓ GPU acceleration is ENABLED and ready!")
            print()
            print("Your system is optimized for massive dataset training.")
            print()
            print("Recommended command:")
            print("  python cli.py reddit dataset.json --batch-size 256")
        else:
            print("⚠ GPU acceleration is NOT available")
            print()
            print("Training will use CPU (slower for large datasets).")
            print()
            print("Options:")
            print("  1. Install CUDA for NVIDIA GPUs")
            print("  2. Use cloud GPU instance (AWS, GCP, Azure)")
            print("  3. Process smaller batches with --max-conversations")
        
        print()
        
    except Exception as e:
        print(f"ERROR: Failed to get GPU information")
        print(f"Details: {e}")
        import traceback
        traceback.print_exc()


def cmd_self_learn(args):
    """Run self-learning training session."""
    print("=" * 70)
    print("SELF-LEARNING TRAINING SESSION")
    print("=" * 70)
    print()
    
    if not os.path.exists(args.database):
        print(f"ERROR: Database '{args.database}' not found!")
        print("Please run 'python cli.py train' first to create the database.")
        return
    
    settings = Settings(
        database_path=args.database,
        spatial_bounds_min=(-500.0, -500.0, -500.0),
        spatial_bounds_max=(500.0, 500.0, 500.0)
    )
    
    container = ApplicationContainer(settings)
    container.initialize()
    
    try:
        language_model = LanguageModel(
            container.graph,
            container.compression_engine,
            container.query_engine,
            container.training_engine,
            enable_self_training=True
        )
        
        # Get initial state
        initial_neurons = len(container.graph.neurons)
        initial_synapses = len(container.graph.synapses)
        
        print(f"Initial state:")
        print(f"  Neurons:  {initial_neurons:,}")
        print(f"  Synapses: {initial_synapses:,}")
        print()
        
        # Training questions
        training_questions = [
            "What are you?",
            "Who are you?",
            "What is your name?",
            "Tell me about yourself",
            "What can you do?",
            "How can you help me?",
            "What is AI?",
            "Explain machine learning",
            "What is a neural network?",
            "How does learning work?",
            "What is intelligence?",
            "Can you think?",
            "Are you conscious?",
            "What is consciousness?",
            "Do you have feelings?",
            "How do you learn?",
            "What makes you intelligent?",
            "Can you understand emotions?",
            "What is your purpose?",
            "How were you created?",
        ]
        
        # Run training rounds
        num_rounds = args.rounds
        
        for round_num in range(1, num_rounds + 1):
            print(f"\n{'=' * 70}")
            print(f"TRAINING ROUND {round_num}/{num_rounds}")
            print(f"{'=' * 70}\n")
            
            # Ask questions
            for i, question in enumerate(training_questions[:args.questions], 1):
                response = language_model.generate_response(
                    question,
                    context_size=5,
                    min_activation=0.1,
                    use_reasoning=True
                )
                
                if args.verbose:
                    print(f"{i}. Q: {question}")
                    print(f"   A: {response[:80]}...")
                else:
                    # Show progress
                    if i % 5 == 0:
                        print(f"  Processed {i}/{len(training_questions[:args.questions])} questions...")
            
            # Consolidate learning
            print(f"\n  Consolidating learning...")
            if language_model._continuous_learning:
                language_model._continuous_learning.self_training.consolidate_learning()
            
            # Show round stats
            if language_model._continuous_learning:
                stats = language_model._continuous_learning.get_statistics()
                print(f"\n  Round {round_num} Results:")
                print(f"    Success rate:      {stats['success_rate']:.1%}")
                print(f"    Positive feedback: {stats['positive_feedback']}")
                print(f"    Negative feedback: {stats['negative_feedback']}")
                print(f"    Neurons removed:   {stats['neurons_removed']}")
                print(f"    Synapses removed:  {stats['synapses_removed']}")
        
        # Final statistics
        final_neurons = len(container.graph.neurons)
        final_synapses = len(container.graph.synapses)
        
        print(f"\n{'=' * 70}")
        print("TRAINING COMPLETE")
        print(f"{'=' * 70}\n")
        
        print(f"Network changes:")
        print(f"  Neurons:  {initial_neurons:,} → {final_neurons:,} ({initial_neurons - final_neurons:+,})")
        print(f"  Synapses: {initial_synapses:,} → {final_synapses:,} ({initial_synapses - final_synapses:+,})")
        print()
        
        if language_model._continuous_learning:
            stats = language_model._continuous_learning.get_statistics()
            print(f"Training statistics:")
            print(f"  Total interactions:    {stats['interaction_count']}")
            print(f"  Success rate:          {stats['success_rate']:.1%}")
            print(f"  Neurons reinforced:    {stats['neurons_reinforced']}")
            print(f"  Neurons weakened:      {stats['neurons_weakened']}")
            print(f"  Neurons removed:       {stats['neurons_removed']}")
            print(f"  Synapses removed:      {stats['synapses_removed']}")
            print(f"  Net quality change:    {stats['net_quality']:+d}")
            print()
        
        # Save changes
        if args.save:
            print("Saving changes to database...")
            if language_model._continuous_learning:
                language_model._continuous_learning.self_training.save_learning_state()
            print("✓ Changes saved")
        else:
            print("Changes NOT saved (use --save to persist)")
        
        print()
        
    finally:
        container.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="F.R.I.D.A.Y AI - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training (first time)
  python cli.py train
  
  # Full training with datasets
  python cli.py train --with-datasets --max-samples 5000
  
  # Incremental update (faster)
  python cli.py update
  
  # Self-learning training (improve AI quality)
  python cli.py learn --rounds 5 --save
  
  # Train from Reddit dataset (with deduplication)
  python cli.py reddit data.json --max-conversations 10000 --save
  
  # Chat with AI
  python cli.py chat
  
  # Test AI
  python cli.py test
  
  # Show statistics
  python cli.py stats
        """
    )
    
    parser.add_argument(
        '--database',
        default='comprehensive_ai.db',
        help='Database file path (default: comprehensive_ai.db)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Full training from scratch')
    train_parser.add_argument('--with-datasets', action='store_true', help='Include external datasets')
    train_parser.add_argument('--max-samples', type=int, default=5000, help='Max samples from datasets')
    train_parser.set_defaults(func=cmd_train_full)
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Incremental update (faster)')
    update_parser.set_defaults(func=cmd_train_incremental)
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat')
    chat_parser.add_argument('--context-size', type=int, default=5, help='Context size')
    chat_parser.add_argument('--min-activation', type=float, default=0.1, help='Min activation threshold')
    chat_parser.set_defaults(func=cmd_chat)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test with predefined questions')
    test_parser.set_defaults(func=cmd_test)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show AI statistics')
    stats_parser.set_defaults(func=cmd_stats)
    
    # Self-learning command
    learn_parser = subparsers.add_parser('learn', help='Run self-learning training')
    learn_parser.add_argument('--rounds', type=int, default=3, help='Number of training rounds (default: 3)')
    learn_parser.add_argument('--questions', type=int, default=20, help='Questions per round (default: 20)')
    learn_parser.add_argument('--save', action='store_true', help='Save changes to database')
    learn_parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    learn_parser.set_defaults(func=cmd_self_learn)
    
    # Reddit training command
    reddit_parser = subparsers.add_parser('reddit', help='Train from Reddit dataset')
    reddit_parser.add_argument('file', help='Path to Reddit JSON file')
    reddit_parser.add_argument('--max-conversations', type=int, default=None, help='Max conversations to load')
    reddit_parser.add_argument('--min-score', type=int, default=2, help='Minimum Reddit score (default: 2)')
    reddit_parser.add_argument('--checkpoint', type=int, default=1000, help='Save checkpoint every N conversations (default: 1000)')
    reddit_parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing (default: 1000)')
    reddit_parser.add_argument('--quality-threshold', type=float, default=0.5, help='Quality threshold (default: 0.5)')
    reddit_parser.add_argument('--similarity-threshold', type=float, default=0.95, help='Similarity threshold for duplicates (default: 0.95)')
    reddit_parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration (use CPU only)')
    reddit_parser.add_argument('--save', action='store_true', help='Save changes to database')
    reddit_parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    reddit_parser.set_defaults(func=cmd_train_reddit)
    
    # GPU info command
    gpu_parser = subparsers.add_parser('gpu-info', help='Show GPU information and capabilities')
    gpu_parser.add_argument('--test', action='store_true', help='Run performance test')
    gpu_parser.set_defaults(func=cmd_gpu_info)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
