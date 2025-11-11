"""
Unified Training Manager for Friday.

Handles all training operations from different data sources.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datasets import load_dataset

from neuron_system.core.graph import NeuronGraph
from neuron_system.engines.compression import CompressionEngine
from neuron_system.engines.query import QueryEngine
from neuron_system.engines.training import TrainingEngine
from neuron_system.storage.database import DatabaseManager
from neuron_system.ai.language_model import LanguageModel

logger = logging.getLogger(__name__)


class TrainingManager:
    """
    Centralized training manager for Friday.
    
    Handles training from:
    - Conversation datasets
    - Q&A datasets
    - Text datasets
    - Custom data
    """
    
    def __init__(self, database_path: str = "data/neuron_system.db"):
        """
        Initialize training manager.
        
        Args:
            database_path: Path to database file
        """
        self.database_path = database_path
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the neuron system."""
        logger.info("Initializing Friday training system...")
        
        # Setup database
        db_path = Path(self.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.db_manager = DatabaseManager(self.database_path)
        self.graph = NeuronGraph()
        self.graph.attach_storage(self.db_manager)
        
        self.compression_engine = CompressionEngine(model_name='all-MiniLM-L6-v2')
        self.query_engine = QueryEngine(self.graph, self.compression_engine)
        self.training_engine = TrainingEngine(self.graph)
        
        # Load existing neurons
        self.graph.load()
        logger.info(f"Loaded {len(self.graph.neurons)} existing neurons")
        
        # Initialize language model
        self.language_model = LanguageModel(
            self.graph,
            self.compression_engine,
            self.query_engine,
            self.training_engine,
            enable_self_training=False
        )
    
    def train_conversations(
        self,
        dataset_name: str = "shihyunlim/english-conversation",
        max_samples: int = 3000,
        min_length: int = 15,
        max_length: int = 1000
    ) -> Dict[str, int]:
        """
        Train from conversation dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            max_samples: Maximum number of samples to train
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Statistics dict with trained/skipped counts
        """
        logger.info(f"Training conversations from {dataset_name}")
        
        # Load dataset
        try:
            dataset = load_dataset(dataset_name, split="train")
            logger.info(f"Loaded {len(dataset)} conversations")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return {"trained": 0, "skipped": 0, "errors": 1}
        
        # Train
        trained = 0
        skipped = 0
        errors = 0
        
        samples_to_train = min(len(dataset), max_samples)
        
        for i in range(samples_to_train):
            try:
                item = dataset[i]
                text = item.get('en', '') if isinstance(item, dict) else str(item)
                
                # Validate
                if not text or len(text) < min_length or len(text) > max_length:
                    skipped += 1
                    continue
                
                # Train
                tags = ['conversation', 'dialogue', 'english', 'natural', 'chat']
                self.language_model.learn(text=text, tags=tags)
                trained += 1
                
                # Progress
                if (i + 1) % 100 == 0:
                    logger.info(f"Progress: {i+1}/{samples_to_train} - Trained: {trained}")
            
            except Exception as e:
                logger.debug(f"Error training sample {i+1}: {e}")
                errors += 1
        
        # Save
        logger.info("Saving to database...")
        self.graph.save()
        
        return {
            "trained": trained,
            "skipped": skipped,
            "errors": errors,
            "total_neurons": len(self.graph.neurons)
        }
    
    def train_qa_pairs(
        self,
        qa_data: List[Dict[str, str]],
        tags: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Train from Q&A pairs.
        
        Args:
            qa_data: List of dicts with 'question' and 'answer' keys
            tags: Optional tags to add
            
        Returns:
            Statistics dict
        """
        logger.info(f"Training {len(qa_data)} Q&A pairs")
        
        if tags is None:
            tags = ['qa', 'knowledge', 'factual']
        
        trained = 0
        skipped = 0
        
        for item in qa_data:
            try:
                question = item.get('question', '').strip()
                answer = item.get('answer', '').strip()
                
                if not question or not answer:
                    skipped += 1
                    continue
                
                # Format as Q&A
                text = f"Question: {question}\nAnswer: {answer}"
                
                # Train
                self.language_model.learn(text=text, tags=tags)
                trained += 1
            
            except Exception as e:
                logger.debug(f"Error training Q&A: {e}")
                skipped += 1
        
        # Save
        self.graph.save()
        
        return {
            "trained": trained,
            "skipped": skipped,
            "total_neurons": len(self.graph.neurons)
        }
    
    def train_text_corpus(
        self,
        texts: List[str],
        tags: Optional[List[str]] = None,
        chunk_size: int = 500
    ) -> Dict[str, int]:
        """
        Train from text corpus.
        
        Args:
            texts: List of text strings
            tags: Optional tags
            chunk_size: Maximum chunk size for long texts
            
        Returns:
            Statistics dict
        """
        logger.info(f"Training {len(texts)} text samples")
        
        if tags is None:
            tags = ['text', 'knowledge']
        
        trained = 0
        skipped = 0
        
        for text in texts:
            try:
                if not text or len(text) < 10:
                    skipped += 1
                    continue
                
                # Chunk if too long
                if len(text) > chunk_size:
                    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                    for chunk in chunks:
                        if len(chunk) > 50:
                            self.language_model.learn(text=chunk, tags=tags)
                            trained += 1
                else:
                    self.language_model.learn(text=text, tags=tags)
                    trained += 1
            
            except Exception as e:
                logger.debug(f"Error training text: {e}")
                skipped += 1
        
        # Save
        self.graph.save()
        
        return {
            "trained": trained,
            "skipped": skipped,
            "total_neurons": len(self.graph.neurons)
        }
    
    def train_reasoning_instructions(
        self,
        instructions: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Train reasoning instructions.
        
        Args:
            instructions: List of instruction dicts
            
        Returns:
            Statistics dict
        """
        logger.info(f"Training {len(instructions)} reasoning instructions")
        
        trained = 0
        
        for instruction in instructions:
            try:
                text = instruction.get('text', '')
                tags = instruction.get('tags', ['reasoning', 'instruction'])
                
                if text:
                    self.language_model.learn(text=text, tags=tags)
                    trained += 1
            
            except Exception as e:
                logger.debug(f"Error training instruction: {e}")
        
        # Save
        self.graph.save()
        
        return {
            "trained": trained,
            "total_neurons": len(self.graph.neurons)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "total_neurons": len(self.graph.neurons),
            "database_path": self.database_path
        }
