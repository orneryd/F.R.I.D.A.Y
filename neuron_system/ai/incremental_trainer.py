"""
Incremental Training System for the AI.

This system allows updating existing knowledge without retraining everything:
- Updates existing neurons with better data
- Adds only new knowledge
- Avoids duplicates
- Much faster than full retraining
"""

import logging
from typing import List, Dict, Any, Optional, Set
from uuid import UUID
import numpy as np

logger = logging.getLogger(__name__)


class IncrementalTrainer:
    """
    Handles incremental training and updates to the AI.
    """
    
    def __init__(self, language_model):
        """
        Initialize incremental trainer.
        
        Args:
            language_model: LanguageModel instance
        """
        self.language_model = language_model
        self.existing_texts: Set[str] = set()
        self.text_to_neuron: Dict[str, UUID] = {}
        
        # Build index of existing knowledge
        self._build_knowledge_index()
    
    def _build_knowledge_index(self):
        """Build an index of existing knowledge for fast lookup."""
        logger.info("Building knowledge index...")
        
        for neuron in self.language_model.graph.neurons.values():
            if hasattr(neuron, 'source_data') and neuron.source_data:
                text = neuron.source_data.strip()
                self.existing_texts.add(text)
                self.text_to_neuron[text] = neuron.id
        
        logger.info(f"Indexed {len(self.existing_texts)} existing knowledge items")
    
    def add_or_update_knowledge(
        self,
        text: str,
        tags: Optional[List[str]] = None,
        force_update: bool = False
    ) -> tuple[UUID, bool]:
        """
        Add new knowledge or update existing if better.
        
        Args:
            text: Knowledge text
            tags: Semantic tags
            force_update: Force update even if text exists
            
        Returns:
            Tuple of (neuron_id, was_updated)
        """
        text = text.strip()
        
        # Check if this exact text already exists
        if text in self.existing_texts and not force_update:
            logger.debug(f"Knowledge already exists, skipping: {text[:50]}...")
            return self.text_to_neuron[text], False
        
        # Check for similar knowledge (fuzzy matching)
        similar_neuron = self._find_similar_knowledge(text, threshold=0.95)
        
        if similar_neuron and not force_update:
            # Update existing neuron with new tags if provided
            if tags:
                existing_tags = getattr(similar_neuron, 'semantic_tags', [])
                new_tags = list(set(existing_tags + tags))
                similar_neuron.semantic_tags = new_tags
                logger.debug(f"Updated tags for existing knowledge: {text[:50]}...")
            return similar_neuron.id, True
        
        # Add as new knowledge
        neuron_id = self.language_model.learn(
            text=text,
            tags=tags,
            create_connections=True
        )
        
        # Update index
        self.existing_texts.add(text)
        self.text_to_neuron[text] = neuron_id
        
        logger.debug(f"Added new knowledge: {text[:50]}...")
        return neuron_id, False
    
    def _find_similar_knowledge(
        self,
        text: str,
        threshold: float = 0.95
    ) -> Optional[Any]:
        """
        Find similar existing knowledge using semantic similarity.
        
        Args:
            text: Text to search for
            threshold: Similarity threshold (0-1)
            
        Returns:
            Similar neuron if found, None otherwise
        """
        # Use query engine to find similar neurons
        results = self.language_model.query_engine.query(
            query_text=text,
            top_k=1,
            propagation_depth=0
        )
        
        if results and results[0].activation >= threshold:
            return results[0].neuron
        
        return None
    
    def batch_add_or_update(
        self,
        knowledge_items: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Add or update multiple knowledge items in batch.
        
        Args:
            knowledge_items: List of dicts with 'text' and optional 'tags'
            batch_size: Progress reporting interval
            
        Returns:
            Statistics dict with counts
        """
        logger.info(f"Processing {len(knowledge_items)} knowledge items...")
        
        stats = {
            'added': 0,
            'updated': 0,
            'skipped': 0
        }
        
        for i, item in enumerate(knowledge_items):
            text = item.get('text', '')
            tags = item.get('tags', [])
            
            if not text or len(text) < 10:
                stats['skipped'] += 1
                continue
            
            neuron_id, was_updated = self.add_or_update_knowledge(text, tags)
            
            if was_updated:
                stats['updated'] += 1
            else:
                stats['added'] += 1
            
            if (i + 1) % batch_size == 0:
                logger.info(f"Processed {i + 1}/{len(knowledge_items)} items...")
        
        logger.info(f"Batch complete: {stats['added']} added, {stats['updated']} updated, {stats['skipped']} skipped")
        return stats
    
    def update_connections(
        self,
        neuron_ids: Optional[List[UUID]] = None,
        top_k: int = 3
    ):
        """
        Update connections for specific neurons or all neurons.
        
        Args:
            neuron_ids: List of neuron IDs to update (None = all)
            top_k: Number of connections per neuron
        """
        if neuron_ids is None:
            neurons = list(self.language_model.graph.neurons.values())
            logger.info(f"Updating connections for all {len(neurons)} neurons...")
        else:
            neurons = [
                self.language_model.graph.neurons[nid]
                for nid in neuron_ids
                if nid in self.language_model.graph.neurons
            ]
            logger.info(f"Updating connections for {len(neurons)} neurons...")
        
        for i, neuron in enumerate(neurons):
            if hasattr(neuron, 'source_data') and neuron.source_data:
                # Remove old connections
                self._remove_neuron_connections(neuron.id)
                
                # Create new connections
                self.language_model._create_connections(neuron, top_k=top_k)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Updated {i + 1}/{len(neurons)} neurons...")
        
        logger.info("Connection update complete")
    
    def _remove_neuron_connections(self, neuron_id: UUID):
        """Remove all synapses connected to a neuron."""
        synapses_to_remove = []
        
        for synapse in self.language_model.graph.synapses.values():
            if synapse.source_neuron_id == neuron_id or synapse.target_neuron_id == neuron_id:
                synapses_to_remove.append(synapse.id)
        
        for synapse_id in synapses_to_remove:
            self.language_model.graph.remove_synapse(synapse_id)
    
    def remove_duplicate_knowledge(self, similarity_threshold: float = 0.98) -> int:
        """
        Remove duplicate or near-duplicate knowledge.
        
        Args:
            similarity_threshold: Threshold for considering items duplicates
            
        Returns:
            Number of duplicates removed
        """
        logger.info("Scanning for duplicate knowledge...")
        
        neurons = list(self.language_model.graph.neurons.values())
        removed_count = 0
        seen_texts = set()
        
        for neuron in neurons:
            if not hasattr(neuron, 'source_data') or not neuron.source_data:
                continue
            
            text = neuron.source_data.strip()
            
            # Check exact duplicates
            if text in seen_texts:
                self.language_model.graph.remove_neuron(neuron.id)
                removed_count += 1
                logger.debug(f"Removed duplicate: {text[:50]}...")
                continue
            
            seen_texts.add(text)
        
        logger.info(f"Removed {removed_count} duplicate knowledge items")
        return removed_count
    
    def save_to_database(self):
        """Save all changes to database."""
        logger.info("Saving incremental updates to database...")
        
        graph = self.language_model.graph
        
        if not hasattr(graph, 'neuron_store') or graph.neuron_store is None:
            logger.warning("No storage attached to graph")
            return
        
        try:
            # Get existing IDs
            existing_neuron_ids = set()
            try:
                existing_neurons = graph.neuron_store.list_all()
                existing_neuron_ids = {n.id for n in existing_neurons if n}
            except Exception as e:
                logger.debug(f"Could not get existing neurons: {e}")
            
            # Separate into create and update
            neurons_to_create = []
            neurons_to_update = []
            
            for n in graph.neurons.values():
                if hasattr(n, 'position') and n.position is not None:
                    if n.id in existing_neuron_ids:
                        neurons_to_update.append(n)
                    else:
                        neurons_to_create.append(n)
            
            # Save neurons
            if neurons_to_create:
                count = graph.neuron_store.batch_create(neurons_to_create)
                logger.info(f"Created {count} new neurons")
            
            if neurons_to_update:
                count = graph.neuron_store.batch_update(neurons_to_update)
                logger.info(f"Updated {count} existing neurons")
            
            # Get existing synapse IDs
            existing_synapse_ids = set()
            try:
                existing_synapses = graph.synapse_store.list_all()
                existing_synapse_ids = {s.id for s in existing_synapses if s}
            except Exception as e:
                logger.debug(f"Could not get existing synapses: {e}")
            
            # Separate synapses
            synapses_to_create = []
            synapses_to_update = []
            
            for s in graph.synapses.values():
                if hasattr(s, 'source_neuron_id') and s.source_neuron_id is not None:
                    if s.id in existing_synapse_ids:
                        synapses_to_update.append(s)
                    else:
                        synapses_to_create.append(s)
            
            # Save synapses
            if synapses_to_create:
                count = graph.synapse_store.batch_create(synapses_to_create)
                logger.info(f"Created {count} new synapses")
            
            if synapses_to_update:
                count = graph.synapse_store.batch_update(synapses_to_update)
                logger.info(f"Updated {count} existing synapses")
            
            logger.info("Incremental save complete")
            
        except Exception as e:
            logger.error(f"Failed to save: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            'total_knowledge_items': len(self.existing_texts),
            'total_neurons': len(self.language_model.graph.neurons),
            'total_synapses': len(self.language_model.graph.synapses),
            'average_connectivity': (
                len(self.language_model.graph.synapses) / 
                len(self.language_model.graph.neurons)
                if len(self.language_model.graph.neurons) > 0 else 0
            )
        }
