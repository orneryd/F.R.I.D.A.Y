"""
Neuron positioning logic for spatial organization.
"""

import random
import numpy as np
from typing import List, Optional, Tuple

from neuron_system.core.vector3d import Vector3D
from neuron_system.core.neuron import Neuron


class NeuronPositioner:
    """
    Handles automatic positioning of neurons in 3D space.
    
    Positions neurons based on semantic similarity and spatial constraints.
    """
    
    def __init__(self, bounds: Tuple[Vector3D, Vector3D]):
        """
        Initialize positioner with spatial bounds.
        
        Args:
            bounds: Tuple of (min_bound, max_bound) defining the space
        """
        self.min_bound, self.max_bound = bounds
        random.seed()
    
    def position_random(self) -> Vector3D:
        """
        Generate a random position within bounds.
        
        Returns:
            Random 3D position
        """
        x = random.uniform(self.min_bound.x, self.max_bound.x)
        y = random.uniform(self.min_bound.y, self.max_bound.y)
        z = random.uniform(self.min_bound.z, self.max_bound.z)
        return Vector3D(x, y, z)
    
    def position_near_similar(self, new_neuron: Neuron, 
                             existing_neurons: List[Neuron],
                             k: int = 5,
                             spread: float = 5.0) -> Vector3D:
        """
        Position a neuron near semantically similar neurons.
        
        Uses vector similarity to find related neurons and positions
        the new neuron nearby with some random spread.
        
        Args:
            new_neuron: Neuron to position (must have vector)
            existing_neurons: List of existing neurons to compare against
            k: Number of similar neurons to consider
            spread: Random spread distance from similar neurons
            
        Returns:
            Calculated 3D position
        """
        if not existing_neurons or new_neuron.vector is None:
            return self.position_random()
        
        # Find k most similar neurons based on vector similarity
        similarities = []
        for neuron in existing_neurons:
            if neuron.vector is not None and neuron.position is not None:
                # Cosine similarity
                similarity = self._cosine_similarity(new_neuron.vector, neuron.vector)
                similarities.append((similarity, neuron))
        
        if not similarities:
            return self.position_random()
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Take top k
        top_k = similarities[:min(k, len(similarities))]
        
        # Calculate average position of similar neurons
        avg_x = sum(n.position.x for _, n in top_k) / len(top_k)
        avg_y = sum(n.position.y for _, n in top_k) / len(top_k)
        avg_z = sum(n.position.z for _, n in top_k) / len(top_k)
        
        # Add random spread
        x = avg_x + random.uniform(-spread, spread)
        y = avg_y + random.uniform(-spread, spread)
        z = avg_z + random.uniform(-spread, spread)
        
        # Clamp to bounds
        x = max(self.min_bound.x, min(x, self.max_bound.x))
        y = max(self.min_bound.y, min(y, self.max_bound.y))
        z = max(self.min_bound.z, min(z, self.max_bound.z))
        
        return Vector3D(x, y, z)
    
    def position_in_cluster(self, cluster_center: Vector3D, 
                           cluster_radius: float) -> Vector3D:
        """
        Position a neuron within a specific cluster region.
        
        Args:
            cluster_center: Center of the cluster
            cluster_radius: Radius of the cluster
            
        Returns:
            Position within the cluster
        """
        # Generate random position within sphere
        # Use rejection sampling for uniform distribution
        while True:
            x = random.uniform(-cluster_radius, cluster_radius)
            y = random.uniform(-cluster_radius, cluster_radius)
            z = random.uniform(-cluster_radius, cluster_radius)
            
            if x*x + y*y + z*z <= cluster_radius * cluster_radius:
                break
        
        # Offset by cluster center
        pos = Vector3D(
            cluster_center.x + x,
            cluster_center.y + y,
            cluster_center.z + z
        )
        
        # Clamp to bounds
        pos.x = max(self.min_bound.x, min(pos.x, self.max_bound.x))
        pos.y = max(self.min_bound.y, min(pos.y, self.max_bound.y))
        pos.z = max(self.min_bound.z, min(pos.z, self.max_bound.z))
        
        return pos
    
    def position_grid(self, index: int, grid_size: int = 10) -> Vector3D:
        """
        Position neurons on a regular grid.
        
        Useful for initial population or testing.
        
        Args:
            index: Index of the neuron
            grid_size: Number of neurons per dimension
            
        Returns:
            Grid position
        """
        # Calculate grid coordinates
        x_idx = index % grid_size
        y_idx = (index // grid_size) % grid_size
        z_idx = index // (grid_size * grid_size)
        
        # Map to bounds
        x_range = self.max_bound.x - self.min_bound.x
        y_range = self.max_bound.y - self.min_bound.y
        z_range = self.max_bound.z - self.min_bound.z
        
        x = self.min_bound.x + (x_idx / grid_size) * x_range
        y = self.min_bound.y + (y_idx / grid_size) * y_range
        z = self.min_bound.z + (z_idx / grid_size) * z_range
        
        return Vector3D(x, y, z)
    
    def rebalance_positions(self, neurons: List[Neuron], 
                           iterations: int = 10,
                           repulsion_strength: float = 1.0) -> dict:
        """
        Rebalance neuron positions to reduce clustering.
        
        Uses a simple force-directed algorithm to spread out neurons
        while maintaining semantic proximity.
        
        Args:
            neurons: List of neurons to rebalance
            iterations: Number of iterations to run
            repulsion_strength: Strength of repulsion force
            
        Returns:
            Dictionary mapping neuron IDs to new positions
        """
        if not neurons:
            return {}
        
        # Create position map
        positions = {n.id: n.position for n in neurons if n.position}
        
        for _ in range(iterations):
            forces = {nid: Vector3D(0, 0, 0) for nid in positions}
            
            # Calculate repulsion forces between all pairs
            neuron_list = list(positions.items())
            for i, (id1, pos1) in enumerate(neuron_list):
                for id2, pos2 in neuron_list[i+1:]:
                    # Calculate repulsion
                    diff = pos1 - pos2
                    dist = pos1.distance(pos2)
                    
                    if dist < 0.1:  # Avoid division by zero
                        dist = 0.1
                    
                    # Repulsion force inversely proportional to distance
                    force_magnitude = repulsion_strength / (dist * dist)
                    force_direction = diff * (1.0 / dist)  # Normalize
                    force = force_direction * force_magnitude
                    
                    forces[id1] = forces[id1] + force
                    forces[id2] = forces[id2] - force
            
            # Apply forces with damping
            damping = 0.5
            for nid, force in forces.items():
                new_pos = positions[nid] + (force * damping)
                
                # Clamp to bounds
                new_pos.x = max(self.min_bound.x, min(new_pos.x, self.max_bound.x))
                new_pos.y = max(self.min_bound.y, min(new_pos.y, self.max_bound.y))
                new_pos.z = max(self.min_bound.z, min(new_pos.z, self.max_bound.z))
                
                positions[nid] = new_pos
        
        return positions
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
