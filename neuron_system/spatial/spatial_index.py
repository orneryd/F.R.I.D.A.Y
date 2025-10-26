"""
SpatialIndex wrapper for managing 3D spatial queries.
"""

from typing import List, Tuple, Optional
from uuid import UUID

from neuron_system.core.vector3d import Vector3D
from neuron_system.core.neuron import Neuron
from neuron_system.spatial.octree import Octree


class SpatialIndex:
    """
    Wrapper class for spatial indexing operations.
    
    Provides a high-level interface for spatial queries and manages
    the underlying octree structure.
    """
    
    def __init__(self, bounds: Tuple[Vector3D, Vector3D], 
                 max_neurons_per_node: int = 8, 
                 max_depth: int = 10):
        """
        Initialize spatial index.
        
        Args:
            bounds: Tuple of (min_bound, max_bound) defining the space
            max_neurons_per_node: Maximum neurons per octree node before subdivision
            max_depth: Maximum octree depth
        """
        self.bounds = bounds
        self.octree = Octree(bounds, max_neurons_per_node, max_depth)
        self._neuron_positions: dict[UUID, Vector3D] = {}
        self._density_threshold = 100  # Neurons per cubic unit before rebalancing
    
    def insert(self, neuron: Neuron) -> bool:
        """
        Insert a neuron into the spatial index.
        
        Args:
            neuron: Neuron to insert
            
        Returns:
            True if inserted successfully, False if out of bounds or no position
        """
        if not neuron.position:
            return False
        
        success = self.octree.insert(neuron)
        if success:
            self._neuron_positions[neuron.id] = neuron.position
        return success
    
    def remove(self, neuron_id: UUID) -> bool:
        """
        Remove a neuron from the spatial index.
        
        Note: This requires rebuilding the octree for efficiency.
        For now, we mark it as removed and rebuild on next rebalance.
        
        Args:
            neuron_id: UUID of neuron to remove
            
        Returns:
            True if neuron was tracked, False otherwise
        """
        if neuron_id in self._neuron_positions:
            del self._neuron_positions[neuron_id]
            return True
        return False
    
    def query_radius(self, center: Vector3D, radius: float) -> List[Neuron]:
        """
        Find all neurons within a given radius of a center point.
        
        Args:
            center: Center point for the query
            radius: Search radius
            
        Returns:
            List of neurons within the radius
        """
        return self.octree.query_radius(center, radius)
    
    def query_knn(self, point: Vector3D, k: int) -> List[Neuron]:
        """
        Find k nearest neighbors to a point.
        
        Args:
            point: Query point
            k: Number of neighbors to find
            
        Returns:
            List of k nearest neurons
        """
        return self.octree.query_knn(point, k)
    
    def query_region(self, min_bound: Vector3D, max_bound: Vector3D) -> List[Neuron]:
        """
        Find all neurons within a rectangular region.
        
        Args:
            min_bound: Minimum corner of the region
            max_bound: Maximum corner of the region
            
        Returns:
            List of neurons within the region
        """
        # Calculate center and radius for sphere query
        center = Vector3D(
            (min_bound.x + max_bound.x) / 2,
            (min_bound.y + max_bound.y) / 2,
            (min_bound.z + max_bound.z) / 2
        )
        
        # Use diagonal as radius to ensure we cover the entire box
        radius = center.distance(max_bound)
        
        # Query with sphere and filter to box
        candidates = self.octree.query_radius(center, radius)
        
        results = []
        for neuron in candidates:
            pos = neuron.position
            if (min_bound.x <= pos.x <= max_bound.x and
                min_bound.y <= pos.y <= max_bound.y and
                min_bound.z <= pos.z <= max_bound.z):
                results.append(neuron)
        
        return results
    
    def get_density(self, center: Vector3D, radius: float) -> float:
        """
        Calculate neuron density in a region.
        
        Args:
            center: Center of the region
            radius: Radius of the spherical region
            
        Returns:
            Density (neurons per cubic unit)
        """
        neurons = self.query_radius(center, radius)
        volume = (4/3) * 3.14159 * (radius ** 3)
        return len(neurons) / volume if volume > 0 else 0
    
    def needs_rebalancing(self) -> bool:
        """
        Check if the spatial index needs rebalancing.
        
        Returns:
            True if rebalancing is recommended
        """
        # Simple heuristic: check if we have removed many neurons
        tracked_count = len(self._neuron_positions)
        indexed_count = self.octree.get_neuron_count()
        
        # If more than 20% discrepancy, rebalance
        if indexed_count > 0:
            discrepancy = abs(tracked_count - indexed_count) / indexed_count
            return discrepancy > 0.2
        
        return False
    
    def rebuild(self, neurons: List[Neuron]):
        """
        Rebuild the spatial index from scratch.
        
        Args:
            neurons: List of all neurons to index
        """
        self.octree.clear()
        self._neuron_positions.clear()
        
        for neuron in neurons:
            self.insert(neuron)
    
    def clear(self):
        """Remove all neurons from the spatial index."""
        self.octree.clear()
        self._neuron_positions.clear()
    
    def get_neuron_count(self) -> int:
        """Get total number of neurons in the index."""
        return self.octree.get_neuron_count()
