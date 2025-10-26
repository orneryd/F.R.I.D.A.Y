"""
Main entry point for the 3D Synaptic Neuron System.

This module provides the main application entry point with proper dependency
injection, configuration management, and lifecycle handling.
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Optional

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuron_system.config.settings import Settings, get_settings, update_settings
from neuron_system.utils.logging_config import setup_logging


# ============================================================================
# Application Container (Dependency Injection)
# ============================================================================

class ApplicationContainer:
    """
    Dependency injection container for the neuron system.
    
    Manages initialization and lifecycle of all system components.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize application container.
        
        Args:
            settings: Optional settings instance (uses defaults if not provided)
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Core components (initialized lazily)
        self._graph = None
        self._compression_engine = None
        self._query_engine = None
        self._training_engine = None
        self._database = None
        self._neuron_store = None
        self._synapse_store = None
        self._spatial_index = None
        
        self._initialized = False
    
    def initialize(self):
        """Initialize all components in correct order."""
        if self._initialized:
            self.logger.warning("Application already initialized")
            return
        
        self.logger.info("Initializing 3D Synaptic Neuron System...")
        
        try:
            # 1. Initialize database
            self._init_database()
            
            # 2. Initialize stores
            self._init_stores()
            
            # 3. Initialize neuron graph
            self._init_graph()
            
            # 4. Load existing data
            self._load_existing_data()
            
            # 5. Initialize engines
            self._init_engines()
            
            self._initialized = True
            self.logger.info("System initialization complete!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}", exc_info=True)
            raise
    
    def _init_database(self):
        """Initialize database connection."""
        from neuron_system.storage.database import DatabaseManager
        
        self.logger.info(f"Initializing database: {self.settings.database_path}")
        self._database = DatabaseManager(self.settings.database_path)
        self.logger.info("Database initialized")
    
    def _init_stores(self):
        """Initialize neuron and synapse stores."""
        from neuron_system.storage.neuron_store import NeuronStore
        from neuron_system.storage.synapse_store import SynapseStore
        
        self.logger.info("Initializing stores...")
        self._neuron_store = NeuronStore(self._database)
        self._synapse_store = SynapseStore(self._database)
        self.logger.info("Stores initialized")
    
    def _init_graph(self):
        """Initialize neuron graph with spatial bounds."""
        from neuron_system.core.graph import NeuronGraph
        from neuron_system.core.vector3d import Vector3D
        
        self.logger.info("Initializing neuron graph...")
        min_bound = Vector3D(*self.settings.spatial_bounds_min)
        max_bound = Vector3D(*self.settings.spatial_bounds_max)
        
        self._graph = NeuronGraph(bounds=(min_bound, max_bound))
        
        # Attach storage for automatic persistence
        self._graph.attach_storage(
            neuron_store=self._neuron_store,
            synapse_store=self._synapse_store
        )
        
        # Use graph's spatial index
        self._spatial_index = self._graph.spatial_index
        
        self.logger.info("Neuron graph initialized")
    
    def _load_existing_data(self):
        """Load existing neurons and synapses from database."""
        self.logger.info("Loading existing data from database...")
        
        # Load neurons
        neurons = self._neuron_store.list_all()
        loaded_neurons = 0
        for neuron in neurons:
            if neuron:
                self._graph.add_neuron(neuron)
                loaded_neurons += 1
        
        self.logger.info(f"Loaded {loaded_neurons} neurons")
        
        # Load synapses
        synapses = self._synapse_store.list_all()
        loaded_synapses = 0
        for synapse in synapses:
            if synapse:
                self._graph.add_synapse(synapse)
                loaded_synapses += 1
        
        self.logger.info(f"Loaded {loaded_synapses} synapses")
    
    def _init_engines(self):
        """Initialize processing engines."""
        from neuron_system.engines.compression import CompressionEngine
        from neuron_system.engines.query import QueryEngine
        from neuron_system.engines.training import TrainingEngine
        
        self.logger.info("Initializing engines...")
        
        # Compression engine
        self._compression_engine = CompressionEngine(
            model_name=self.settings.embedding_model
        )
        self.logger.info("Compression engine initialized")
        
        # Query engine
        self._query_engine = QueryEngine(
            neuron_graph=self._graph,
            compression_engine=self._compression_engine
        )
        self.logger.info("Query engine initialized")
        
        # Training engine
        self._training_engine = TrainingEngine(
            neuron_graph=self._graph
        )
        self.logger.info("Training engine initialized")
    
    def shutdown(self):
        """Cleanup and shutdown all components."""
        if not self._initialized:
            return
        
        self.logger.info("Shutting down 3D Synaptic Neuron System...")
        
        try:
            # Save all pending changes
            # Note: The graph's storage attachment handles automatic persistence,
            # but we can explicitly update here if needed
            if self._neuron_store and self._graph:
                # Only save neurons with valid positions
                neurons_to_save = [
                    n for n in self._graph.neurons.values()
                    if hasattr(n, 'position') and n.position is not None
                ]
                if neurons_to_save:
                    self._neuron_store.batch_update(neurons_to_save)
                self.logger.info(f"Saved {len(neurons_to_save)} neurons")
            
            if self._synapse_store and self._graph:
                # Only save synapses with valid data
                synapses_to_save = [
                    s for s in self._graph.synapses.values()
                    if hasattr(s, 'source_neuron_id') and s.source_neuron_id is not None
                ]
                if synapses_to_save:
                    self._synapse_store.batch_update(synapses_to_save)
                self.logger.info(f"Saved {len(synapses_to_save)} synapses")
            
            # Close database
            if self._database:
                self._database.close()
                self.logger.info("Database connection closed")
            
            self._initialized = False
            self.logger.info("Shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)
    
    # Property accessors for components
    @property
    def graph(self):
        """Get neuron graph instance."""
        if not self._initialized:
            raise RuntimeError("Application not initialized")
        return self._graph
    
    @property
    def compression_engine(self):
        """Get compression engine instance."""
        if not self._initialized:
            raise RuntimeError("Application not initialized")
        return self._compression_engine
    
    @property
    def query_engine(self):
        """Get query engine instance."""
        if not self._initialized:
            raise RuntimeError("Application not initialized")
        return self._query_engine
    
    @property
    def training_engine(self):
        """Get training engine instance."""
        if not self._initialized:
            raise RuntimeError("Application not initialized")
        return self._training_engine
    
    @property
    def database(self):
        """Get database manager instance."""
        if not self._initialized:
            raise RuntimeError("Application not initialized")
        return self._database
    
    @property
    def neuron_store(self):
        """Get neuron store instance."""
        if not self._initialized:
            raise RuntimeError("Application not initialized")
        return self._neuron_store
    
    @property
    def synapse_store(self):
        """Get synapse store instance."""
        if not self._initialized:
            raise RuntimeError("Application not initialized")
        return self._synapse_store
    
    @property
    def spatial_index(self):
        """Get spatial index instance."""
        if not self._initialized:
            raise RuntimeError("Application not initialized")
        return self._spatial_index


# ============================================================================
# Global Application Instance
# ============================================================================

_app_container: Optional[ApplicationContainer] = None


def get_app_container() -> ApplicationContainer:
    """
    Get the global application container instance.
    
    Returns:
        ApplicationContainer instance
    """
    global _app_container
    if _app_container is None:
        _app_container = ApplicationContainer()
    return _app_container


def initialize_app(settings: Optional[Settings] = None):
    """
    Initialize the application with optional custom settings.
    
    Args:
        settings: Optional settings instance
    """
    global _app_container
    _app_container = ApplicationContainer(settings)
    _app_container.initialize()


def shutdown_app():
    """Shutdown the application."""
    global _app_container
    if _app_container:
        _app_container.shutdown()
        _app_container = None


# ============================================================================
# CLI Interface
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="3D Synaptic Neuron System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start API server
  python main.py api
  
  # Start API server on custom port
  python main.py api --port 8080
  
  # Run interactive shell
  python main.py shell
  
  # Show system status
  python main.py status
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--database",
        type=str,
        help="Path to database file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Start API server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Shell command
    subparsers.add_parser("shell", help="Start interactive shell")
    
    # Status command
    subparsers.add_parser("status", help="Show system status")
    
    # Health check command
    subparsers.add_parser("health", help="Check system health")
    
    return parser


def run_api_server(args):
    """Run the API server."""
    import uvicorn
    
    print("=" * 70)
    print("3D Synaptic Neuron System API Server")
    print("=" * 70)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    print(f"ReDoc: http://{args.host}:{args.port}/redoc")
    print("=" * 70)
    print("Press CTRL+C to stop the server")
    print()
    
    uvicorn.run(
        "neuron_system.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower()
    )


def run_shell(args):
    """Run interactive shell."""
    print("=" * 70)
    print("3D Synaptic Neuron System - Interactive Shell")
    print("=" * 70)
    print()
    
    # Initialize application
    initialize_app()
    container = get_app_container()
    
    print(f"Loaded {len(container.graph.neurons)} neurons")
    print(f"Loaded {len(container.graph.synapses)} synapses")
    print()
    print("Available objects:")
    print("  - container: Application container")
    print("  - graph: Neuron graph")
    print("  - query_engine: Query engine")
    print("  - training_engine: Training engine")
    print()
    
    # Start interactive shell
    import code
    code.interact(
        local={
            "container": container,
            "graph": container.graph,
            "query_engine": container.query_engine,
            "training_engine": container.training_engine,
        },
        banner=""
    )
    
    # Cleanup
    shutdown_app()


def run_status(args):
    """Show system status."""
    print("=" * 70)
    print("3D Synaptic Neuron System - Status")
    print("=" * 70)
    print()
    
    # Initialize application
    initialize_app()
    container = get_app_container()
    
    print(f"Database: {container.settings.database_path}")
    print(f"Neurons: {len(container.graph.neurons)}")
    print(f"Synapses: {len(container.graph.synapses)}")
    print(f"Spatial bounds: {container.settings.spatial_bounds_min} to {container.settings.spatial_bounds_max}")
    print(f"Embedding model: {container.settings.embedding_model}")
    print()
    
    # Cleanup
    shutdown_app()


def run_health_check(args):
    """Run health check."""
    print("=" * 70)
    print("3D Synaptic Neuron System - Health Check")
    print("=" * 70)
    print()
    
    try:
        # Initialize application
        initialize_app()
        container = get_app_container()
        
        # Check components
        checks = {
            "Database": container.database is not None,
            "Neuron Store": container.neuron_store is not None,
            "Synapse Store": container.synapse_store is not None,
            "Neuron Graph": container.graph is not None,
            "Spatial Index": container.spatial_index is not None,
            "Compression Engine": container.compression_engine is not None,
            "Query Engine": container.query_engine is not None,
            "Training Engine": container.training_engine is not None,
        }
        
        all_healthy = True
        for component, status in checks.items():
            status_str = "✓ OK" if status else "✗ FAILED"
            print(f"{component:.<40} {status_str}")
            if not status:
                all_healthy = False
        
        print()
        if all_healthy:
            print("Status: HEALTHY")
            exit_code = 0
        else:
            print("Status: UNHEALTHY")
            exit_code = 1
        
        # Cleanup
        shutdown_app()
        
        return exit_code
        
    except Exception as e:
        print(f"Health check failed: {e}")
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Update settings from CLI args
    if args.database:
        update_settings(database_path=args.database)
    
    # Execute command
    if args.command == "api":
        run_api_server(args)
    elif args.command == "shell":
        run_shell(args)
    elif args.command == "status":
        run_status(args)
    elif args.command == "health":
        exit_code = run_health_check(args)
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
