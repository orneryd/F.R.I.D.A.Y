"""
F.R.I.D.A.Y AI - Command Line Interface

Modernes CLI mit Dimension-Support und allen Features.
"""

import argparse
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def get_model_config(dimension: str = 'auto'):
    """
    Get model configuration for specified dimension.
    
    Args:
        dimension: 'auto', '384', '768', or '1024'
        
    Returns:
        Tuple of (sentence_transformer_model, pretrained_model, dimension)
    """
    configs = {
        '384': ('all-MiniLM-L6-v2', 'distilbert-base-uncased', 384),
        '768': ('all-mpnet-base-v2', 'bert-base-uncased', 768),
        '1024': ('sentence-transformers/all-roberta-large-v1', 'roberta-large', 1024),
    }
    
    if dimension == 'auto':
        # Default to 384D
        return configs['384']
    
    if dimension not in configs:
        logger.error(f"Unknown dimension: {dimension}. Use: auto, 384, 768, or 1024")
        sys.exit(1)
    
    return configs[dimension]


def init_system(database: str, dimension: str = 'auto', use_neural: bool = False):
    """
    Initialize the neuron system.
    
    Args:
        database: Database path
        dimension: Dimension to use
        use_neural: Whether to use neural inference
        
    Returns:
        Tuple of (graph, compression_engine, query_engine, training_engine, language_model)
    """
    from neuron_system.core.graph import NeuronGraph
    from neuron_system.engines.compression import CompressionEngine
    from neuron_system.engines.query import QueryEngine
    from neuron_system.engines.training import TrainingEngine
    from neuron_system.storage.database import DatabaseManager
    
    # Ensure data directory exists
    db_path = Path(database)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get model config
    sentence_model, pretrained_model, expected_dim = get_model_config(dimension)
    
    logger.info(f"Initializing system with {expected_dim}D...")
    logger.info(f"Model: {sentence_model}")
    
    # Initialize components
    db_manager = DatabaseManager(database)
    graph = NeuronGraph()
    graph.attach_storage(db_manager)
    
    compression_engine = CompressionEngine(model_name=sentence_model)
    query_engine = QueryEngine(graph, compression_engine)
    training_engine = TrainingEngine(graph)
    
    # Load existing data
    try:
        graph.load()
        logger.info(f"Loaded {len(graph.neurons)} neurons from database")
    except Exception as e:
        logger.debug(f"No existing data: {e}")
    
    # Initialize language model
    if use_neural:
        try:
            from neuron_system.ai.smart_language_model import SmartLanguageModel
            language_model = SmartLanguageModel(
                graph, compression_engine, query_engine, training_engine,
                pretrained_model=pretrained_model,
                enable_self_training=True
            )
            logger.info("Neural Inference enabled")
        except Exception as e:
            logger.warning(f"Neural Inference not available: {e}")
            from neuron_system.ai.language_model import LanguageModel
            language_model = LanguageModel(
                graph, compression_engine, query_engine, training_engine,
                enable_self_training=True
            )
    else:
        from neuron_system.ai.language_model import LanguageModel
        language_model = LanguageModel(
            graph, compression_engine, query_engine, training_engine,
            enable_self_training=True
        )
    
    # Verify dimension
    compression_engine._ensure_model_loaded()
    actual_dim = compression_engine.vector_dim
    logger.info(f"System ready: {actual_dim}D")
    
    if actual_dim != expected_dim:
        logger.warning(f"Dimension mismatch: expected {expected_dim}D, got {actual_dim}D")
    
    return graph, compression_engine, query_engine, training_engine, language_model


# ============================================================================
# COMMANDS
# ============================================================================

def cmd_train(args):
    """Train the AI with dataset."""
    logger.info("=" * 70)
    logger.info("TRAINING")
    logger.info("=" * 70)
    
    graph, _, _, _, language_model = init_system(args.database, args.dimension)
    
    from neuron_system.ai.training import SmartTrainer
    from neuron_system.ai.datasets import DatasetLoader
    
    trainer = SmartTrainer(language_model)
    
    # Load dataset
    try:
        dataset = DatasetLoader.load(args.dataset, format=args.format)
        conversations = list(dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("\nAvailable datasets:")
        DatasetLoader.print_available()
        sys.exit(1)
    
    if not conversations:
        logger.error("No conversations found in dataset!")
        sys.exit(1)
    
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Conversations: {len(conversations)}")
    logger.info("")
    
    # Train
    for i, conv in enumerate(conversations, 1):
        question = conv.get('question', '')
        answer = conv.get('answer', '')
        
        if not question or not answer:
            logger.warning(f"  [{i}/{len(conversations)}] ✗ Invalid conversation")
            continue
        
        success, reason = trainer.train_conversation(question, answer)
        status = "✓" if success else "✗"
        
        if args.verbose or not success:
            logger.info(f"  [{i}/{len(conversations)}] {status} {question[:50]}...")
            if not success:
                logger.info(f"      Reason: {reason}")
        elif i % 10 == 0:
            logger.info(f"  [{i}/{len(conversations)}] Processing...")
    
    # Save
    if hasattr(graph, 'neuron_store') and graph.neuron_store:
        try:
            # Save all neurons
            neurons = list(graph.neurons.values())
            if neurons:
                graph.neuron_store.batch_update(neurons)
            
            # Save all synapses
            synapses = list(graph.synapses.values())
            if synapses:
                graph.synapse_store.batch_update(synapses)
            
            logger.info("Saved to database")
        except Exception as e:
            logger.warning(f"Could not save: {e}")
    
    # Stats
    stats = trainer.get_statistics()
    logger.info("")
    logger.info("Training complete:")
    logger.info(f"  Processed: {stats['total_processed']}")
    logger.info(f"  Added: {stats['successfully_added']}")
    logger.info(f"  Duplicates: {stats['duplicates_found']}")
    logger.info(f"  Rejected: {stats['quality_rejected'] + stats['logic_rejected']}")
    logger.info(f"  Success rate: {stats['success_rate']:.1%}")


def cmd_chat(args):
    """Interactive chat with the AI."""
    logger.info("=" * 70)
    logger.info("CHAT MODE")
    logger.info("=" * 70)
    logger.info("Commands: 'exit' to quit, 'stats' for statistics")
    logger.info("")
    
    _, _, _, _, language_model = init_system(
        args.database, args.dimension, args.neural
    )
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit']:
                logger.info("Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                stats = language_model.get_statistics()
                logger.info("")
                logger.info("Statistics:")
                logger.info(f"  Neurons: {stats['total_neurons']}")
                logger.info(f"  Synapses: {stats['total_synapses']}")
                logger.info("")
                continue
            
            # Generate response
            response = language_model.generate_response(
                user_input,
                context_size=args.context_size,
                use_neural_inference=args.neural
            )
            
            logger.info(f"AI: {response}")
            logger.info("")
            
        except KeyboardInterrupt:
            logger.info("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def cmd_learn(args):
    """Add single knowledge item."""
    logger.info("=" * 70)
    logger.info("LEARNING")
    logger.info("=" * 70)
    
    graph, _, _, _, language_model = init_system(args.database, args.dimension)
    
    tags = args.tags.split(',') if args.tags else []
    
    logger.info(f"Learning: {args.text[:50]}...")
    
    neuron_id = language_model.learn(
        text=args.text,
        tags=tags,
        create_connections=True
    )
    
    graph.save()
    
    logger.info(f"✓ Learned (Neuron ID: {neuron_id})")


def cmd_query(args):
    """Query the AI."""
    logger.info("=" * 70)
    logger.info("QUERY")
    logger.info("=" * 70)
    
    _, _, _, _, language_model = init_system(
        args.database, args.dimension, args.neural
    )
    
    logger.info(f"Query: {args.query}")
    logger.info("")
    
    response = language_model.generate_response(
        args.query,
        context_size=args.top_k,
        use_neural_inference=args.neural
    )
    
    logger.info(f"Response: {response}")


def cmd_stats(args):
    """Show system statistics."""
    logger.info("=" * 70)
    logger.info("STATISTICS")
    logger.info("=" * 70)
    
    graph, compression_engine, _, _, language_model = init_system(
        args.database, args.dimension
    )
    
    stats = language_model.get_statistics()
    
    logger.info(f"Database: {args.database}")
    logger.info(f"Dimension: {compression_engine.vector_dim}D")
    logger.info("")
    logger.info(f"Neurons: {stats['total_neurons']}")
    logger.info(f"  Knowledge: {stats['knowledge_neurons']}")
    logger.info(f"  Memory: {stats['memory_neurons']}")
    logger.info(f"  Tool: {stats['tool_neurons']}")
    logger.info("")
    logger.info(f"Synapses: {stats['total_synapses']}")
    logger.info(f"Avg Connectivity: {stats['average_connectivity']:.2f}")


def cmd_gpu_info(args):
    """Show GPU information."""
    logger.info("=" * 70)
    logger.info("GPU INFORMATION")
    logger.info("=" * 70)
    
    from neuron_system.engines.gpu_accelerator import GPUAccelerator
    
    gpu = GPUAccelerator()
    info = gpu.get_device_info()
    
    logger.info(f"Device: {info['device_name']}")
    logger.info(f"Type: {info['device_type']}")
    logger.info(f"Available: {info['is_available']}")
    
    if info['is_available']:
        logger.info(f"Memory Total: {info.get('memory_total_gb', 'N/A')} GB")
        logger.info(f"Memory Available: {info.get('memory_available_gb', 'N/A')} GB")
    
    if args.test:
        logger.info("")
        logger.info("Running performance test...")
        # TODO: Add performance test


def cmd_list_datasets(args):
    """List available datasets."""
    from neuron_system.ai.datasets import DatasetLoader
    DatasetLoader.print_available()


def cmd_validate_persistence(args):
    """Validate data persistence."""
    logger.info("=" * 70)
    logger.info("PERSISTENCE VALIDATION")
    logger.info("=" * 70)
    
    # Test database
    test_db = "data/test_persistence.db"
    
    # Remove existing test database
    test_path = Path(test_db)
    if test_path.exists():
        test_path.unlink()
        logger.info("Removed existing test database")
    
    logger.info("")
    logger.info("Phase 1: Create and Save Data")
    logger.info("-" * 40)
    
    # Create system manually (without auto-load)
    from neuron_system.storage.database import DatabaseManager
    from neuron_system.core.graph import NeuronGraph
    from neuron_system.engines.compression import CompressionEngine
    from neuron_system.engines.query import QueryEngine
    from neuron_system.engines.training import TrainingEngine
    from neuron_system.ai.language_model import LanguageModel
    
    sentence_model, _, _ = get_model_config(args.dimension)
    
    db1 = DatabaseManager(test_db)
    graph1 = NeuronGraph()
    graph1.attach_storage(db1)
    
    compression_engine1 = CompressionEngine(model_name=sentence_model)
    query_engine1 = QueryEngine(graph1, compression_engine1)
    training_engine1 = TrainingEngine(graph1)
    language_model1 = LanguageModel(
        graph1, compression_engine1, query_engine1, training_engine1,
        enable_self_training=False
    )
    
    # Create test data
    test_data = [
        ("AI is artificial intelligence", ['ai', 'definition']),
        ("ML is machine learning", ['ml', 'definition']),
        ("Python is a programming language", ['python', 'programming']),
    ]
    
    logger.info(f"Creating {len(test_data)} neurons...")
    for text, tags in test_data:
        language_model1.learn(text, tags=tags)
    
    # Save
    stats_before = {'neurons': len(graph1.neurons), 'synapses': len(graph1.synapses)}
    logger.info(f"Before save: {stats_before['neurons']} neurons, {stats_before['synapses']} synapses")
    
    try:
        graph1.save()
        logger.info("✓ Save successful")
    except Exception as e:
        logger.error(f"✗ Save failed: {e}")
        return
    
    logger.info("")
    logger.info("Phase 2: Load and Verify Data")
    logger.info("-" * 40)
    
    # Create new system and load
    db2 = DatabaseManager(test_db)
    graph2 = NeuronGraph()
    graph2.attach_storage(db2)
    
    compression_engine2 = CompressionEngine(model_name=sentence_model)
    query_engine2 = QueryEngine(graph2, compression_engine2)
    training_engine2 = TrainingEngine(graph2)
    language_model2 = LanguageModel(
        graph2, compression_engine2, query_engine2, training_engine2,
        enable_self_training=False
    )
    
    try:
        graph2.load()
        logger.info("✓ Load successful")
    except Exception as e:
        logger.error(f"✗ Load failed: {e}")
        return
    
    # Compare
    stats_after = {'neurons': len(graph2.neurons), 'synapses': len(graph2.synapses)}
    logger.info(f"After load: {stats_after['neurons']} neurons, {stats_after['synapses']} synapses")
    
    logger.info("")
    logger.info("Phase 3: Validation")
    logger.info("-" * 40)
    
    # Validate
    neurons_match = stats_before['neurons'] == stats_after['neurons']
    synapses_match = stats_before['synapses'] == stats_after['synapses']
    
    if neurons_match:
        logger.info(f"✓ Neuron count matches: {stats_after['neurons']}")
    else:
        logger.error(f"✗ Neuron count mismatch: {stats_before['neurons']} -> {stats_after['neurons']}")
    
    if synapses_match:
        logger.info(f"✓ Synapse count matches: {stats_after['synapses']}")
    else:
        logger.error(f"✗ Synapse count mismatch: {stats_before['synapses']} -> {stats_after['synapses']}")
    
    # Functional test
    try:
        response = language_model2.generate_response("What is AI?")
        if response and len(response) > 10:
            logger.info("✓ System functional after load")
        else:
            logger.warning("System functional but weak response")
    except Exception as e:
        logger.error(f"✗ System not functional: {e}")
    
    # Summary
    logger.info("")
    if neurons_match and synapses_match:
        logger.info("✅ Persistence validation PASSED")
        logger.info(f"Database: {test_db}")
    else:
        logger.info("❌ Persistence validation FAILED")


def cmd_validate_3d(args):
    """Validate 3D system."""
    logger.info("=" * 70)
    logger.info("3D SYSTEM VALIDATION")
    logger.info("=" * 70)
    
    graph, _, _, _, language_model = init_system(args.database, args.dimension)
    
    # Create test neurons if empty
    if len(graph.neurons) == 0:
        logger.info("Creating test neurons for validation...")
        
        test_data = [
            ("AI is artificial intelligence", ['ai', 'definition']),
            ("ML is machine learning", ['ml', 'definition']),
            ("DL is deep learning", ['dl', 'definition']),
            ("NLP is natural language processing", ['nlp', 'definition']),
            ("Python is a programming language", ['python', 'programming']),
        ]
        
        for text, tags in test_data:
            language_model.learn(text, tags=tags)
        
        logger.info(f"Created {len(test_data)} test neurons")
    
    logger.info("")
    
    # Validation checks
    neurons_without_position = sum(1 for n in graph.neurons.values() if n.position is None)
    neurons_without_vector = sum(1 for n in graph.neurons.values() if n.vector is None)
    
    # Position bounds check
    min_bound, max_bound = graph.bounds
    out_of_bounds = 0
    for neuron in graph.neurons.values():
        if neuron.position:
            if not (min_bound.x <= neuron.position.x <= max_bound.x and
                    min_bound.y <= neuron.position.y <= max_bound.y and
                    min_bound.z <= neuron.position.z <= max_bound.z):
                out_of_bounds += 1
    
    # Synapse validation
    invalid_synapses = 0
    for synapse in graph.synapses.values():
        if (synapse.source_neuron_id not in graph.neurons or
            synapse.target_neuron_id not in graph.neurons):
            invalid_synapses += 1
    
    # Results
    logger.info(f"Neurons: {len(graph.neurons)}")
    logger.info(f"  With positions: {len(graph.neurons) - neurons_without_position}")
    logger.info(f"  With vectors: {len(graph.neurons) - neurons_without_vector}")
    logger.info(f"  Within bounds: {len(graph.neurons) - out_of_bounds}")
    logger.info("")
    logger.info(f"Synapses: {len(graph.synapses)}")
    logger.info(f"  Valid: {len(graph.synapses) - invalid_synapses}")
    if len(graph.neurons) > 0:
        avg_connectivity = len(graph.synapses) / len(graph.neurons)
        logger.info(f"  Avg connectivity: {avg_connectivity:.2f}")
    
    # Spatial index test
    logger.info("")
    try:
        if graph.neurons:
            test_neuron = list(graph.neurons.values())[0]
            nearby = graph.spatial_index.query_radius(test_neuron.position, radius=20.0)
            logger.info(f"Spatial index: Working ({len(nearby)} nearby neurons found)")
        else:
            logger.info("Spatial index: No neurons to test")
    except Exception as e:
        logger.error(f"Spatial index: Error - {e}")
    
    # Summary
    all_good = (
        neurons_without_position == 0 and
        neurons_without_vector == 0 and
        out_of_bounds == 0 and
        invalid_synapses == 0
    )
    
    logger.info("")
    if all_good:
        logger.info("✅ 3D System validation PASSED")
    else:
        logger.info("❌ 3D System validation FAILED")


def cmd_cluster(args):
    """Cluster neurons for better organization."""
    logger.info("=" * 70)
    logger.info("NEURON CLUSTERING")
    logger.info("=" * 70)
    
    graph, _, _, _, _ = init_system(args.database, args.dimension)
    
    if len(graph.neurons) == 0:
        logger.warning("No neurons in database. Train the AI first.")
        return
    
    logger.info(f"Clustering {len(graph.neurons)} neurons...")
    logger.info("")
    
    # Import clustering engine
    from neuron_system.spatial.neuron_clustering import NeuronClusteringEngine
    
    engine = NeuronClusteringEngine()
    neurons = list(graph.neurons.values())
    
    # Choose clustering method
    method = args.method
    
    if method == 'kmeans':
        logger.info(f"Using K-Means clustering (k={args.k})...")
        clusters = engine.cluster_kmeans(neurons, n_clusters=args.k)
    elif method == 'dbscan':
        logger.info(f"Using DBSCAN clustering (eps={args.eps}, min_samples={args.min_samples})...")
        clusters = engine.cluster_dbscan(neurons, eps=args.eps, min_samples=args.min_samples)
    elif method == 'topics':
        logger.info("Using topic-based clustering...")
        clusters = engine.cluster_by_topics(neurons)
    elif method == 'hybrid':
        logger.info(f"Using hybrid clustering (k={args.k})...")
        clusters = engine.cluster_hybrid(neurons, n_clusters=args.k)
    else:
        logger.error(f"Unknown method: {method}")
        return
    
    logger.info("")
    logger.info("Clustering Results:")
    logger.info("-" * 40)
    
    # Show clusters
    for cluster in sorted(clusters.values(), key=lambda c: c.size(), reverse=True)[:10]:
        tags_str = ', '.join(list(cluster.tags)[:3]) if cluster.tags else 'no tags'
        quality_str = f", quality={cluster.quality_score:.3f}" if cluster.quality_score > 0 else ""
        logger.info(f"  Cluster {cluster.id} ({cluster.name}): {cluster.size()} neurons{quality_str}, tags=[{tags_str}]")
    
    if len(clusters) > 10:
        logger.info(f"  ... and {len(clusters) - 10} more clusters")
    
    # Statistics
    logger.info("")
    stats = engine.get_statistics()
    logger.info("Statistics:")
    logger.info(f"  Total clusters: {stats['num_clusters']}")
    logger.info(f"  Total neurons: {stats['total_neurons']}")
    logger.info(f"  Avg cluster size: {stats['avg_cluster_size']:.1f}")
    logger.info(f"  Min cluster size: {stats['min_cluster_size']}")
    logger.info(f"  Max cluster size: {stats['max_cluster_size']}")
    if stats['avg_quality_score'] > 0:
        logger.info(f"  Avg quality score: {stats['avg_quality_score']:.3f}")
    
    logger.info("")
    logger.info("✅ Clustering complete")


def cmd_migrate(args):
    """Migrate to different dimension."""
    logger.info("=" * 70)
    logger.info("DIMENSION MIGRATION")
    logger.info("=" * 70)
    
    # Load old system
    logger.info(f"Loading from: {args.source}")
    old_graph, _, _, _, old_model = init_system(args.source, 'auto')
    
    # Get old dimension
    old_dim = old_model.compression_engine.vector_dim
    logger.info(f"Source dimension: {old_dim}D")
    
    # Create new system
    logger.info(f"Creating target: {args.target}")
    new_graph, _, _, _, new_model = init_system(args.target, args.dimension)
    
    # Get new dimension
    new_dim = new_model.compression_engine.vector_dim
    logger.info(f"Target dimension: {new_dim}D")
    
    if old_dim == new_dim:
        logger.warning("Same dimension! No migration needed.")
        return
    
    # Migrate neurons
    logger.info("")
    logger.info("Migrating neurons...")
    
    from neuron_system.ai.training import IncrementalTrainer
    trainer = IncrementalTrainer(new_model)
    
    migrated = 0
    for neuron in old_graph.neurons.values():
        if hasattr(neuron, 'source_data') and neuron.source_data:
            tags = getattr(neuron, 'semantic_tags', [])
            trainer.add_or_update_knowledge(neuron.source_data, tags)
            migrated += 1
            
            if migrated % 100 == 0:
                logger.info(f"  Migrated {migrated} neurons...")
    
    # Save
    if hasattr(new_graph, 'neuron_store') and new_graph.neuron_store:
        try:
            neurons = list(new_graph.neurons.values())
            if neurons:
                new_graph.neuron_store.batch_update(neurons)
            synapses = list(new_graph.synapses.values())
            if synapses:
                new_graph.synapse_store.batch_update(synapses)
        except Exception as e:
            logger.warning(f"Could not save: {e}")
    
    logger.info("")
    logger.info(f"✓ Migration complete: {migrated} neurons")
    logger.info(f"  {old_dim}D → {new_dim}D")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='F.R.I.D.A.Y AI - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global options
    parser.add_argument(
        '--database', '-db',
        default='data/neuron_system.db',
        help='Database path (default: data/neuron_system.db)'
    )
    parser.add_argument(
        '--dimension', '-d',
        choices=['auto', '384', '768', '1024'],
        default='auto',
        help='Vector dimension (default: auto = 384D)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the AI')
    train_parser.add_argument(
        '--dataset', '-ds',
        default='basic-ai',
        help='Dataset to use (default: basic-ai). Use "list" to see available datasets.'
    )
    train_parser.add_argument(
        '--format', '-f',
        choices=['json', 'reddit'],
        default='json',
        help='Dataset format for files (default: json)'
    )
    train_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output (show all conversations)'
    )
    train_parser.set_defaults(func=cmd_train)
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat')
    chat_parser.add_argument(
        '--context-size', '-c',
        type=int, default=5,
        help='Number of context neurons (default: 5)'
    )
    chat_parser.add_argument(
        '--neural', '-n',
        action='store_true',
        help='Use neural inference'
    )
    chat_parser.set_defaults(func=cmd_chat)
    
    # Learn command
    learn_parser = subparsers.add_parser('learn', help='Add knowledge')
    learn_parser.add_argument('text', help='Text to learn')
    learn_parser.add_argument('--tags', '-t', help='Tags (comma-separated)')
    learn_parser.set_defaults(func=cmd_learn)
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the AI')
    query_parser.add_argument('query', help='Query text')
    query_parser.add_argument(
        '--top-k', '-k',
        type=int, default=5,
        help='Number of results (default: 5)'
    )
    query_parser.add_argument(
        '--neural', '-n',
        action='store_true',
        help='Use neural inference'
    )
    query_parser.set_defaults(func=cmd_query)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    stats_parser.set_defaults(func=cmd_stats)
    
    # GPU info command
    gpu_parser = subparsers.add_parser('gpu-info', help='GPU information')
    gpu_parser.add_argument('--test', action='store_true', help='Run performance test')
    gpu_parser.set_defaults(func=cmd_gpu_info)
    
    # List datasets command
    list_ds_parser = subparsers.add_parser('list-datasets', help='List available datasets')
    list_ds_parser.set_defaults(func=cmd_list_datasets)
    
    # Validate persistence command
    validate_persist_parser = subparsers.add_parser('validate-persistence', help='Validate data persistence')
    validate_persist_parser.set_defaults(func=cmd_validate_persistence)
    
    # Validate 3D command
    validate_3d_parser = subparsers.add_parser('validate-3d', help='Validate 3D system')
    validate_3d_parser.set_defaults(func=cmd_validate_3d)
    
    # Cluster command
    cluster_parser = subparsers.add_parser('cluster', help='Cluster neurons')
    cluster_parser.add_argument(
        '--method', '-m',
        choices=['kmeans', 'dbscan', 'topics', 'hybrid'],
        default='hybrid',
        help='Clustering method (default: hybrid)'
    )
    cluster_parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of clusters for k-means/hybrid (default: 10)'
    )
    cluster_parser.add_argument(
        '--eps',
        type=float,
        default=0.3,
        help='DBSCAN epsilon parameter (default: 0.3)'
    )
    cluster_parser.add_argument(
        '--min-samples',
        type=int,
        default=3,
        help='DBSCAN min_samples parameter (default: 3)'
    )
    cluster_parser.set_defaults(func=cmd_cluster)
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate to different dimension')
    migrate_parser.add_argument('source', help='Source database')
    migrate_parser.add_argument('target', help='Target database')
    migrate_parser.set_defaults(func=cmd_migrate)
    
    # Parse args
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
