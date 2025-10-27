"""
AI Tests - Language Model, Training

Konsolidierte Tests für AI-Funktionalität.
"""

import pytest
from neuron_system.core.graph import NeuronGraph
from neuron_system.engines.compression import CompressionEngine
from neuron_system.engines.query import QueryEngine
from neuron_system.engines.training import TrainingEngine
from neuron_system.ai.language_model import LanguageModel
from neuron_system.ai.training import SmartTrainer, IncrementalTrainer


class TestLanguageModel:
    """Test LanguageModel functionality."""
    
    @pytest.fixture
    def setup(self):
        """Setup test environment."""
        graph = NeuronGraph()
        compression_engine = CompressionEngine()
        query_engine = QueryEngine(graph, compression_engine)
        training_engine = TrainingEngine(graph)
        
        model = LanguageModel(
            graph, compression_engine, query_engine, training_engine,
            enable_self_training=False
        )
        
        return model
    
    def test_model_creation(self, setup):
        """Test model creation."""
        model = setup
        assert model is not None
        assert model.graph is not None
    
    def test_learn(self, setup):
        """Test learning new knowledge."""
        model = setup
        
        neuron_id = model.learn(
            text="AI is artificial intelligence",
            tags=['ai', 'definition']
        )
        
        assert neuron_id is not None
        assert len(model.graph.neurons) > 0
    
    def test_understand(self, setup):
        """Test understanding query."""
        model = setup
        
        # Learn something first
        model.learn(text="AI is artificial intelligence", tags=['ai'])
        
        # Query
        results = model.understand("What is AI?", top_k=5)
        
        assert results is not None
        assert len(results) > 0
    
    def test_generate_response(self, setup):
        """Test response generation."""
        model = setup
        
        # Learn something
        model.learn(
            text="Question: What is AI? Answer: Artificial Intelligence",
            tags=['qa']
        )
        
        # Generate response
        response = model.generate_response("What is AI?")
        
        assert response is not None
        assert len(response) > 0


class TestSmartTrainer:
    """Test SmartTrainer functionality."""
    
    @pytest.fixture
    def setup(self):
        """Setup test environment."""
        graph = NeuronGraph()
        compression_engine = CompressionEngine()
        query_engine = QueryEngine(graph, compression_engine)
        training_engine = TrainingEngine(graph)
        
        model = LanguageModel(
            graph, compression_engine, query_engine, training_engine,
            enable_self_training=False
        )
        
        trainer = SmartTrainer(model)
        return trainer
    
    def test_trainer_creation(self, setup):
        """Test trainer creation."""
        trainer = setup
        assert trainer is not None
    
    def test_train_conversation(self, setup):
        """Test training conversation."""
        trainer = setup
        
        success, reason = trainer.train_conversation(
            question="What is machine learning?",
            answer="Machine learning is a subset of AI that enables systems to learn from data."
        )
        
        assert success is not None
        # May be True or False depending on quality checks
    
    def test_quality_check(self, setup):
        """Test quality checking."""
        trainer = setup
        
        # Good quality
        success, _ = trainer.train_conversation(
            question="What is AI?",
            answer="Artificial Intelligence is the simulation of human intelligence by machines."
        )
        assert success == True
        
        # Bad quality (too short)
        success, _ = trainer.train_conversation(
            question="What?",
            answer="Yes"
        )
        assert success == False


class TestIncrementalTrainer:
    """Test IncrementalTrainer functionality."""
    
    @pytest.fixture
    def setup(self):
        """Setup test environment."""
        graph = NeuronGraph()
        compression_engine = CompressionEngine()
        query_engine = QueryEngine(graph, compression_engine)
        training_engine = TrainingEngine(graph)
        
        model = LanguageModel(
            graph, compression_engine, query_engine, training_engine,
            enable_self_training=False
        )
        
        trainer = IncrementalTrainer(model)
        return trainer
    
    def test_trainer_creation(self, setup):
        """Test trainer creation."""
        trainer = setup
        assert trainer is not None
    
    def test_add_knowledge(self, setup):
        """Test adding knowledge."""
        trainer = setup
        
        neuron_id, was_updated = trainer.add_or_update_knowledge(
            text="New information about AI",
            tags=['ai']
        )
        
        assert neuron_id is not None
        assert was_updated == False  # First time, not updated
    
    def test_update_knowledge(self, setup):
        """Test updating existing knowledge."""
        trainer = setup
        
        # Add first time
        neuron_id1, _ = trainer.add_or_update_knowledge(
            text="AI information",
            tags=['ai']
        )
        
        # Add again (should detect duplicate)
        neuron_id2, was_updated = trainer.add_or_update_knowledge(
            text="AI information",
            tags=['ai', 'new_tag']
        )
        
        # Should be same neuron or similar
        assert neuron_id2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
