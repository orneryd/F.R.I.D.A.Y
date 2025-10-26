"""
Dataset loader for training the neuron system with real-world data.

Supports loading from various sources like:
- HuggingFace datasets
- Wikipedia
- Common Crawl
- Text files
- JSON files
"""

import logging
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads datasets from various sources for training the neuron system.
    """
    
    def __init__(self, language_model):
        """
        Initialize dataset loader.
        
        Args:
            language_model: LanguageModel instance to load data into
        """
        self.language_model = language_model
        self.loaded_count = 0
    
    def load_from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        text_field: str = "text",
        max_samples: Optional[int] = None,
        batch_size: int = 100
    ) -> int:
        """
        Load dataset from HuggingFace.
        
        Args:
            dataset_name: Name of the dataset (e.g., "wikipedia", "bookcorpus")
            split: Dataset split to use (train, test, validation)
            text_field: Name of the text field in the dataset
            max_samples: Maximum number of samples to load (None = all)
            batch_size: Batch size for progress logging
            
        Returns:
            Number of neurons created
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed. Install with: pip install datasets")
            return 0
        
        logger.info(f"Loading dataset: {dataset_name} (split: {split})")
        
        try:
            # Try to load with config if needed
            try:
                dataset = load_dataset(dataset_name, split=split, streaming=True)
            except ValueError as e:
                # If config is needed, try common configs
                if "wikitext" in dataset_name.lower():
                    logger.info("Trying wikitext-2-raw-v1 config...")
                    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split, streaming=True)
                else:
                    raise
            
            count = 0
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                
                text = item.get(text_field, "")
                if not text or len(text) < 10:  # Skip very short texts
                    continue
                
                # Extract tags from dataset metadata
                tags = self._extract_tags(item, dataset_name)
                
                # Learn the text
                self.language_model.learn(
                    text=text,
                    tags=tags,
                    create_connections=False  # Create connections in batch later
                )
                
                count += 1
                self.loaded_count += 1
                
                if count % batch_size == 0:
                    logger.info(f"Loaded {count} samples from {dataset_name}...")
            
            logger.info(f"Completed loading {count} samples from {dataset_name}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            return 0
    
    def load_wikipedia(
        self,
        language: str = "en",
        max_articles: Optional[int] = 1000,
        batch_size: int = 50
    ) -> int:
        """
        Load Wikipedia articles.
        
        Args:
            language: Language code (en, de, fr, etc.)
            max_articles: Maximum number of articles to load
            batch_size: Batch size for progress logging
            
        Returns:
            Number of neurons created
        """
        logger.info(f"Loading Wikipedia articles (language: {language})")
        
        # Use the 20220301 version which works without scripts
        return self.load_from_huggingface(
            dataset_name=f"wikipedia",
            split=f"20220301.{language}",
            text_field="text",
            max_samples=max_articles,
            batch_size=batch_size
        )
    
    def load_from_text_file(
        self,
        file_path: str,
        chunk_size: int = 500,
        overlap: int = 50,
        tags: Optional[List[str]] = None
    ) -> int:
        """
        Load text from a file and split into chunks.
        
        Args:
            file_path: Path to text file
            chunk_size: Size of text chunks (in characters)
            overlap: Overlap between chunks
            tags: Tags to apply to all chunks
            
        Returns:
            Number of neurons created
        """
        logger.info(f"Loading text file: {file_path}")
        
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return 0
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split into chunks
            chunks = self._split_text(text, chunk_size, overlap)
            
            count = 0
            for chunk in chunks:
                if len(chunk.strip()) < 10:
                    continue
                
                self.language_model.learn(
                    text=chunk,
                    tags=tags or ["file", path.stem],
                    create_connections=False
                )
                count += 1
                self.loaded_count += 1
            
            logger.info(f"Loaded {count} chunks from {file_path}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            return 0
    
    def load_from_json(
        self,
        file_path: str,
        text_field: str = "text",
        tags_field: Optional[str] = "tags",
        max_items: Optional[int] = None
    ) -> int:
        """
        Load data from JSON file.
        
        JSON format:
        [
            {"text": "...", "tags": ["tag1", "tag2"]},
            ...
        ]
        
        Args:
            file_path: Path to JSON file
            text_field: Field name containing text
            tags_field: Field name containing tags (optional)
            max_items: Maximum number of items to load
            
        Returns:
            Number of neurons created
        """
        logger.info(f"Loading JSON file: {file_path}")
        
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return 0
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.error("JSON must contain a list of items")
                return 0
            
            count = 0
            for i, item in enumerate(data):
                if max_items and i >= max_items:
                    break
                
                text = item.get(text_field, "")
                if not text or len(text) < 10:
                    continue
                
                tags = item.get(tags_field, []) if tags_field else []
                
                self.language_model.learn(
                    text=text,
                    tags=tags,
                    create_connections=False
                )
                count += 1
                self.loaded_count += 1
            
            logger.info(f"Loaded {count} items from {file_path}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            return 0
    
    def load_common_crawl_sample(
        self,
        max_samples: int = 1000,
        batch_size: int = 100
    ) -> int:
        """
        Load a sample from Common Crawl dataset.
        
        Args:
            max_samples: Maximum number of samples to load
            batch_size: Batch size for progress logging
            
        Returns:
            Number of neurons created
        """
        logger.info("Loading Common Crawl sample...")
        
        return self.load_from_huggingface(
            dataset_name="c4",  # Colossal Clean Crawled Corpus
            split="train",
            text_field="text",
            max_samples=max_samples,
            batch_size=batch_size
        )
    
    def load_bookcorpus(
        self,
        max_samples: int = 1000,
        batch_size: int = 100
    ) -> int:
        """
        Load BookCorpus dataset.
        
        Args:
            max_samples: Maximum number of samples to load
            batch_size: Batch size for progress logging
            
        Returns:
            Number of neurons created
        """
        logger.info("Loading BookCorpus...")
        
        return self.load_from_huggingface(
            dataset_name="bookcorpus",
            split="train",
            text_field="text",
            max_samples=max_samples,
            batch_size=batch_size
        )
    
    def create_connections_batch(self, top_k: int = 3, save_interval: int = 1000, max_neurons: int = None):
        """
        Create connections between loaded neurons in batch.
        
        This is more efficient than creating connections one by one.
        Also saves progress periodically to database.
        
        Args:
            top_k: Number of connections per neuron
            save_interval: Save to database every N neurons (0 = no intermediate saves)
            max_neurons: Maximum number of neurons to process (None = all)
        """
        logger.info("Creating connections between neurons...")
        
        neurons = list(self.language_model.graph.neurons.values())
        total = len(neurons)
        
        # Limit to max_neurons if specified
        if max_neurons and max_neurons < total:
            neurons = neurons[:max_neurons]
            print(f"   Processing {len(neurons)} of {total} neurons (optimized for speed)...")
        else:
            print(f"   Processing {total} neurons...")
        
        process_count = len(neurons)
        
        for i, neuron in enumerate(neurons):
            if hasattr(neuron, 'source_data') and neuron.source_data:
                self.language_model._create_connections(neuron, top_k=top_k)
            
            # Periodic save
            if save_interval > 0 and (i + 1) % save_interval == 0:
                self._save_progress()
                print(f"   Progress: {i + 1}/{process_count} neurons ({(i+1)*100//process_count}%) - Saved to database")
            
            # Progress indicator every 50 neurons
            if (i + 1) % 50 == 0:
                print(f"   Progress: {i + 1}/{process_count} neurons ({(i+1)*100//process_count}%)")
        
        # Final save
        self._save_progress()
        logger.info(f"Completed creating connections for {process_count} neurons")
    
    def _save_progress(self):
        """Save current neurons and synapses to database."""
        graph = self.language_model.graph
        
        # Check if storage is attached
        if not hasattr(graph, 'neuron_store') or graph.neuron_store is None:
            logger.warning("No storage attached to graph, skipping save")
            return
        
        try:
            # Get existing neuron IDs from database
            existing_ids = set()
            try:
                existing_neurons = graph.neuron_store.list_all()
                existing_ids = {n.id for n in existing_neurons if n}
            except Exception as e:
                logger.debug(f"Could not get existing neurons: {e}")
            
            # Separate neurons into create and update
            neurons_to_create = []
            neurons_to_update = []
            
            for n in graph.neurons.values():
                if hasattr(n, 'position') and n.position is not None:
                    if n.id in existing_ids:
                        neurons_to_update.append(n)
                    else:
                        neurons_to_create.append(n)
            
            # Create new neurons
            if neurons_to_create:
                count = graph.neuron_store.batch_create(neurons_to_create)
                logger.debug(f"Created {count} new neurons")
            
            # Update existing neurons
            if neurons_to_update:
                count = graph.neuron_store.batch_update(neurons_to_update)
                logger.debug(f"Updated {count} existing neurons")
            
            # Get existing synapse IDs from database
            existing_synapse_ids = set()
            try:
                existing_synapses = graph.synapse_store.list_all()
                existing_synapse_ids = {s.id for s in existing_synapses if s}
            except Exception as e:
                logger.debug(f"Could not get existing synapses: {e}")
            
            # Separate synapses into create and update
            synapses_to_create = []
            synapses_to_update = []
            
            for s in graph.synapses.values():
                if hasattr(s, 'source_neuron_id') and s.source_neuron_id is not None:
                    if s.id in existing_synapse_ids:
                        synapses_to_update.append(s)
                    else:
                        synapses_to_create.append(s)
            
            # Create new synapses
            if synapses_to_create:
                count = graph.synapse_store.batch_create(synapses_to_create)
                logger.debug(f"Created {count} new synapses")
            
            # Update existing synapses
            if synapses_to_update:
                count = graph.synapse_store.batch_update(synapses_to_update)
                logger.debug(f"Updated {count} existing synapses")
                
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def _split_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:  # Only break if we're past halfway
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def _extract_tags(self, item: Dict[str, Any], dataset_name: str) -> List[str]:
        """
        Extract tags from dataset item.
        
        Args:
            item: Dataset item
            dataset_name: Name of the dataset
            
        Returns:
            List of tags
        """
        tags = [dataset_name]
        
        # Add common metadata fields as tags
        for field in ['category', 'topic', 'domain', 'language']:
            if field in item:
                tags.append(str(item[field]))
        
        return tags
    
    def save_to_database(self):
        """
        Explicitly save all neurons and synapses to database.
        
        This is called automatically during training, but can be called
        manually to ensure everything is persisted.
        """
        logger.info("Saving all data to database...")
        self._save_progress()
        logger.info("Save complete")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get loading statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_loaded': self.loaded_count,
            'neurons_in_graph': len(self.language_model.graph.neurons),
            'synapses_in_graph': len(self.language_model.graph.synapses)
        }
