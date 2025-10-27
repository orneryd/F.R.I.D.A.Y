"""
Smart Positioning - Intelligente 3D-Positionierung für Neuronen.

Verbesserte Positionierungs-Strategien:
- Semantic Clustering (ähnliche Neuronen nah beieinander)
- Density-aware Positioning (vermeidet Überfüllung)
- Topic-based Regions (verschiedene Topics in verschiedenen Regionen)
"""

import logging
import numpy as np
from typing import List, Optional, Dict, Any
from collections import defaultdict

from neuron_system.core.vector3d import Vector3D
from neuron_system.core.neuron import Neuron

logger = logging.getLogger(__name__)


class SmartPositioner:
    """
    Intelligente Positionierung für Neuronen im 3D-Raum.
    """
    
    def __init__(self, bounds: tuple):
        """
        Initialize SmartPositioner.
        
        Args:
            bounds: Tuple of (min_bound, max_bound)
        """
        self.min_bound, self.max_bound = bounds
        
        # Topic regions (verschiedene Topics in verschiedenen Bereichen)
        self.topic_regions = {}
        self.next_region_index = 0
        
        # Density tracking
        self.density_grid = {}
        self.grid_size = 10.0  # Grid cell size
    
    def position_semantic(
        self,
        neuron: Neuron,
        existing_neurons: List[Neuron],
        top_k: int = 5,
        spread: float = 5.0
    ) -> Vector3D:
        """
        Position neuron near semantically similar neurons.
        
        Args:
            neuron: Neuron to position
            existing_neurons: Existing neurons in graph
            top_k: Number of similar neurons to consider
            spread: Distance spread around similar neurons
            
        Returns:
            Position vector
        """
        if not existing_neurons or neuron.vector is None:
            return self._position_random()
        
        # Find similar neurons
        similarities = []
        for existing in existing_neurons:
            if existing.vector is not None and len(existing.vector) == len(neuron.vector):
                # Cosine similarity
                sim = np.dot(neuron.vector, existing.vector) / (
                    np.linalg.norm(neuron.vector) * np.linalg.norm(existing.vector) + 1e-8
                )
                if existing.position:
                    similarities.append((sim, existing.position))
        
        if not similarities:
            return self._position_random()
        
        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Take top-k
        top_positions = [pos for _, pos in similarities[:top_k]]
        
        # Calculate centroid
        centroid = Vector3D(
            sum(p.x for p in top_positions) / len(top_positions),
            sum(p.y for p in top_positions) / len(top_positions),
            sum(p.z for p in top_positions) / len(top_positions)
        )
        
        # Add random spread
        offset = Vector3D(
            np.random.normal(0, spread),
            np.random.normal(0, spread),
            np.random.normal(0, spread)
        )
        
        position = Vector3D(
            centroid.x + offset.x,
            centroid.y + offset.y,
            centroid.z + offset.z
        )
        
        # Clamp to bounds
        return self._clamp_to_bounds(position)
    
    def position_by_topic(
        self,
        tags: List[str],
        spread: float = 10.0
    ) -> Vector3D:
        """
        Position neuron in topic-specific region.
        
        Args:
            tags: Semantic tags
            spread: Spread within region
            
        Returns:
            Position vector
        """
        if not tags:
            return self._position_random()
        
        # Use first tag as primary topic
        primary_topic = tags[0]
        
        # Get or create region for this topic
        if primary_topic not in self.topic_regions:
            self.topic_regions[primary_topic] = self._allocate_region()
        
        region_center = self.topic_regions[primary_topic]
        
        # Random position within region
        offset = Vector3D(
            np.random.normal(0, spread),
            np.random.normal(0, spread),
            np.random.normal(0, spread)
        )
        
        position = Vector3D(
            region_center.x + offset.x,
            region_center.y + offset.y,
            region_center.z + offset.z
        )
        
        return self._clamp_to_bounds(position)
    
    def position_density_aware(
        self,
        existing_neurons: List[Neuron],
        min_distance: float = 5.0
    ) -> Vector3D:
        """
        Position neuron avoiding high-density areas.
        
        Args:
            existing_neurons: Existing neurons
            min_distance: Minimum distance to other neurons
            
        Returns:
            Position vector
        """
        # Update density grid
        self._update_density_grid(existing_neurons)
        
        # Try to find low-density position
        max_attempts = 100
        for _ in range(max_attempts):
            candidate = self._position_random()
            
            # Check density at this position
            grid_key = self._get_grid_key(candidate)
            density = self.density_grid.get(grid_key, 0)
            
            # If low density, check actual distances
            if density < 5:  # Max 5 neurons per grid cell
                # Check minimum distance
                too_close = False
                for existing in existing_neurons:
                    if existing.position:
                        dist = candidate.distance_to(existing.position)
                        if dist < min_distance:
                            too_close = True
                            break
                
                if not too_close:
                    return candidate
        
        # Fallback: random position
        return self._position_random()
    
    def _allocate_region(self) -> Vector3D:
        """
        Allocate a new region for a topic.
        
        Returns:
            Region center
        """
        # Divide space into regions (simple grid)
        regions_per_axis = 5
        
        # Calculate region index
        x_idx = self.next_region_index % regions_per_axis
        y_idx = (self.next_region_index // regions_per_axis) % regions_per_axis
        z_idx = (self.next_region_index // (regions_per_axis ** 2)) % regions_per_axis
        
        self.next_region_index += 1
        
        # Calculate region center
        x_range = self.max_bound.x - self.min_bound.x
        y_range = self.max_bound.y - self.min_bound.y
        z_range = self.max_bound.z - self.min_bound.z
        
        x = self.min_bound.x + (x_idx + 0.5) * (x_range / regions_per_axis)
        y = self.min_bound.y + (y_idx + 0.5) * (y_range / regions_per_axis)
        z = self.min_bound.z + (z_idx + 0.5) * (z_range / regions_per_axis)
        
        return Vector3D(x, y, z)
    
    def _update_density_grid(self, neurons: List[Neuron]):
        """Update density grid with current neurons."""
        self.density_grid.clear()
        
        for neuron in neurons:
            if neuron.position:
                grid_key = self._get_grid_key(neuron.position)
                self.density_grid[grid_key] = self.density_grid.get(grid_key, 0) + 1
    
    def _get_grid_key(self, position: Vector3D) -> tuple:
        """Get grid cell key for position."""
        x_cell = int(position.x / self.grid_size)
        y_cell = int(position.y / self.grid_size)
        z_cell = int(position.z / self.grid_size)
        return (x_cell, y_cell, z_cell)
    
    def _position_random(self) -> Vector3D:
        """Generate random position within bounds."""
        x = np.random.uniform(self.min_bound.x, self.max_bound.x)
        y = np.random.uniform(self.min_bound.y, self.max_bound.y)
        z = np.random.uniform(self.min_bound.z, self.max_bound.z)
        return Vector3D(x, y, z)
    
    def _clamp_to_bounds(self, position: Vector3D) -> Vector3D:
        """Clamp position to bounds."""
        x = np.clip(position.x, self.min_bound.x, self.max_bound.x)
        y = np.clip(position.y, self.min_bound.y, self.max_bound.y)
        z = np.clip(position.z, self.min_bound.z, self.max_bound.z)
        return Vector3D(x, y, z)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get positioning statistics."""
        return {
            'topic_regions': len(self.topic_regions),
            'topics': list(self.topic_regions.keys()),
            'density_cells': len(self.density_grid),
            'max_density': max(self.density_grid.values()) if self.density_grid else 0,
        }


__all__ = ['SmartPositioner']
