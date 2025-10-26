"""
Tests for the main entry point and application container.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ApplicationContainer, initialize_app, shutdown_app, get_app_container
from neuron_system.config.settings import Settings


class TestApplicationContainer:
    """Test the ApplicationContainer class."""
    
    def test_container_initialization(self):
        """Test that container initializes all components."""
        # Create container with test database
        settings = Settings(database_path=":memory:")
        container = ApplicationContainer(settings)
        
        # Initialize
        container.initialize()
        
        # Verify all components are initialized
        assert container.graph is not None
        assert container.compression_engine is not None
        assert container.query_engine is not None
        assert container.training_engine is not None
        assert container.database is not None
        assert container.neuron_store is not None
        assert container.synapse_store is not None
        assert container.spatial_index is not None
        
        # Cleanup
        container.shutdown()
    
    def test_container_lazy_properties(self):
        """Test that accessing properties before initialization raises error."""
        container = ApplicationContainer()
        
        with pytest.raises(RuntimeError, match="Application not initialized"):
            _ = container.graph
    
    def test_container_double_initialization(self):
        """Test that double initialization is handled gracefully."""
        settings = Settings(database_path=":memory:")
        container = ApplicationContainer(settings)
        
        container.initialize()
        container.initialize()  # Should log warning but not fail
        
        container.shutdown()
    
    def test_global_app_container(self):
        """Test global application container functions."""
        # Initialize
        settings = Settings(database_path=":memory:")
        initialize_app(settings)
        
        # Get container
        container = get_app_container()
        assert container is not None
        assert container.graph is not None
        
        # Shutdown
        shutdown_app()
        
        # After shutdown, get_app_container should return None or new instance
        # (depending on implementation)


class TestComponentIntegration:
    """Test integration between components."""
    
    def test_neuron_creation_and_query(self):
        """Test creating neurons and querying them."""
        settings = Settings(database_path=":memory:")
        container = ApplicationContainer(settings)
        container.initialize()
        
        try:
            # Create a knowledge neuron
            from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
            from neuron_system.core.vector3d import Vector3D
            import numpy as np
            
            # Compress some text
            vector = container.compression_engine.compress("Python programming language")
            
            neuron = KnowledgeNeuron(
                position=Vector3D(0, 0, 0),
                vector=vector,
                source_data="Python is a high-level programming language",
                compression_ratio=1000.0,
                semantic_tags=["programming", "python"]
            )
            
            # Add to graph
            container.graph.add_neuron(neuron)
            
            # Query
            results = container.query_engine.query("programming", top_k=5)
            
            # Verify results
            assert len(results) > 0
            assert results[0].neuron.id == neuron.id
            
        finally:
            container.shutdown()
    
    def test_training_operations(self):
        """Test training engine operations."""
        settings = Settings(database_path=":memory:")
        container = ApplicationContainer(settings)
        container.initialize()
        
        try:
            # Create a neuron
            from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
            from neuron_system.core.vector3d import Vector3D
            import numpy as np
            
            vector = np.random.randn(384)
            neuron = KnowledgeNeuron(
                position=Vector3D(0, 0, 0),
                vector=vector,
                source_data="Test data",
                compression_ratio=1000.0,
                semantic_tags=["test"]
            )
            
            container.graph.add_neuron(neuron)
            
            # Adjust neuron
            new_vector = np.random.randn(384)
            container.training_engine.adjust_neuron(
                neuron.id,
                new_vector,
                learning_rate=0.5
            )
            
            # Verify adjustment
            updated_neuron = container.graph.get_neuron(neuron.id)
            assert not np.array_equal(updated_neuron.vector, vector)
            
        finally:
            container.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
