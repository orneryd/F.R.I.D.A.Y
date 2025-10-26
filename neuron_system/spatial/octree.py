"""
Octree implementation for efficient 3D spatial queries.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import heapq

from neuron_system.core.vector3d import Vector3D
from neuron_system.core.neuron import Neuron


@dataclass
class OctreeNode:
    """
    A node in the octree structure.
    
    Each node represents a cubic region of 3D space and can contain
    neurons or be subdivided into 8 child octants.
    """
    
    center: Vector3D
    half_size: float
    neurons: List[Neuron]
    children: Optional[List['OctreeNode']]
    
    def __init__(self, center: Vector3D, half_size: float):
        """
        Initialize an octree node.
        
        Args:
            center: Center point of the cubic region
            half_size: Half the size of the cube (distance from center to face)
        """
        self.center = center
        self.half_size = half_size
        self.neurons = []
        self.children = None
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children)."""
        return self.children is None
    
    def contains_point(self, point: Vector3D) -> bool:
        """
        Check if a point is within this node's bounds.
        
        Args:
            point: Point to check
            
        Returns:
            True if point is within bounds
        """
        return (
            abs(point.x - self.center.x) <= self.half_size and
            abs(point.y - self.center.y) <= self.half_size and
            abs(point.z - self.center.z) <= self.half_size
        )
    
    def intersects_sphere(self, center: Vector3D, radius: float) -> bool:
        """
        Check if a sphere intersects this node's bounds.
        
        Args:
            center: Center of the sphere
            radius: Radius of the sphere
            
        Returns:
            True if sphere intersects this node
        """
        # Find the closest point in the box to the sphere center
        closest_x = max(self.center.x - self.half_size, 
                       min(center.x, self.center.x + self.half_size))
        closest_y = max(self.center.y - self.half_size,
                       min(center.y, self.center.y + self.half_size))
        closest_z = max(self.center.z - self.half_size,
                       min(center.z, self.center.z + self.half_size))
        
        closest_point = Vector3D(closest_x, closest_y, closest_z)
        distance = center.distance(closest_point)
        
        return distance <= radius


class Octree:
    """
    Octree data structure for efficient 3D spatial queries.
    
    Supports insertion, radius queries, and k-nearest neighbor search.
    """
    
    def __init__(self, bounds: Tuple[Vector3D, Vector3D], max_neurons_per_node: int = 8, max_depth: int = 10):
        """
        Initialize octree with spatial bounds.
        
        Args:
            bounds: Tuple of (min_bound, max_bound) defining the space
            max_neurons_per_node: Maximum neurons before subdivision
            max_depth: Maximum tree depth to prevent excessive subdivision
        """
        min_bound, max_bound = bounds
        
        # Calculate center and half-size
        center = Vector3D(
            (min_bound.x + max_bound.x) / 2,
            (min_bound.y + max_bound.y) / 2,
            (min_bound.z + max_bound.z) / 2
        )
        
        half_size = max(
            (max_bound.x - min_bound.x) / 2,
            (max_bound.y - min_bound.y) / 2,
            (max_bound.z - min_bound.z) / 2
        )
        
        self.root = OctreeNode(center, half_size)
        self.max_neurons_per_node = max_neurons_per_node
        self.max_depth = max_depth
        self.neuron_count = 0
    
    def insert(self, neuron: Neuron) -> bool:
        """
        Insert a neuron into the octree.
        
        Args:
            neuron: Neuron to insert
            
        Returns:
            True if inserted successfully, False if out of bounds
        """
        if not neuron.position:
            return False
        
        if not self.root.contains_point(neuron.position):
            return False
        
        self._insert_recursive(self.root, neuron, 0)
        self.neuron_count += 1
        return True

    def _insert_recursive(self, node: OctreeNode, neuron: Neuron, depth: int):
        """
        Recursively insert a neuron into the octree.
        
        Args:
            node: Current node
            neuron: Neuron to insert
            depth: Current depth in the tree
        """
        # If this is a leaf node
        if node.is_leaf():
            node.neurons.append(neuron)
            
            # Subdivide if necessary
            if len(node.neurons) > self.max_neurons_per_node and depth < self.max_depth:
                self._subdivide(node)
                
                # Redistribute neurons to children
                neurons_to_redistribute = node.neurons
                node.neurons = []
                
                for n in neurons_to_redistribute:
                    self._insert_into_children(node, n, depth)
        else:
            # Insert into appropriate child
            self._insert_into_children(node, neuron, depth)
    
    def _insert_into_children(self, node: OctreeNode, neuron: Neuron, depth: int):
        """
        Insert neuron into the appropriate child node.
        
        Args:
            node: Parent node with children
            neuron: Neuron to insert
            depth: Current depth
        """
        octant = self._get_octant(node, neuron.position)
        self._insert_recursive(node.children[octant], neuron, depth + 1)
    
    def _subdivide(self, node: OctreeNode):
        """
        Subdivide a node into 8 children.
        
        Args:
            node: Node to subdivide
        """
        half = node.half_size / 2
        cx, cy, cz = node.center.x, node.center.y, node.center.z
        
        # Create 8 child nodes (octants)
        node.children = [
            OctreeNode(Vector3D(cx - half, cy - half, cz - half), half),  # 0: ---
            OctreeNode(Vector3D(cx + half, cy - half, cz - half), half),  # 1: +--
            OctreeNode(Vector3D(cx - half, cy + half, cz - half), half),  # 2: -+-
            OctreeNode(Vector3D(cx + half, cy + half, cz - half), half),  # 3: ++-
            OctreeNode(Vector3D(cx - half, cy - half, cz + half), half),  # 4: --+
            OctreeNode(Vector3D(cx + half, cy - half, cz + half), half),  # 5: +--+
            OctreeNode(Vector3D(cx - half, cy + half, cz + half), half),  # 6: -++
            OctreeNode(Vector3D(cx + half, cy + half, cz + half), half),  # 7: +++
        ]
    
    def _get_octant(self, node: OctreeNode, point: Vector3D) -> int:
        """
        Determine which octant a point belongs to.
        
        Args:
            node: Parent node
            point: Point to classify
            
        Returns:
            Octant index (0-7)
        """
        octant = 0
        if point.x >= node.center.x:
            octant |= 1
        if point.y >= node.center.y:
            octant |= 2
        if point.z >= node.center.z:
            octant |= 4
        return octant
    
    def query_radius(self, center: Vector3D, radius: float) -> List[Neuron]:
        """
        Find all neurons within a given radius of a center point.
        
        Args:
            center: Center point for the query
            radius: Search radius
            
        Returns:
            List of neurons within the radius
        """
        results = []
        self._query_radius_recursive(self.root, center, radius, results)
        return results
    
    def _query_radius_recursive(self, node: OctreeNode, center: Vector3D, 
                                radius: float, results: List[Neuron]):
        """
        Recursively search for neurons within radius.
        
        Args:
            node: Current node
            center: Query center
            radius: Search radius
            results: List to accumulate results
        """
        # Check if this node intersects the search sphere
        if not node.intersects_sphere(center, radius):
            return
        
        # Check neurons in this node
        for neuron in node.neurons:
            if neuron.position.distance(center) <= radius:
                results.append(neuron)
        
        # Recursively check children
        if not node.is_leaf():
            for child in node.children:
                self._query_radius_recursive(child, center, radius, results)
    
    def query_knn(self, point: Vector3D, k: int) -> List[Neuron]:
        """
        Find k nearest neighbors to a point.
        
        Args:
            point: Query point
            k: Number of neighbors to find
            
        Returns:
            List of k nearest neurons (or fewer if not enough neurons exist)
        """
        # Use a max heap to keep track of k nearest neurons
        # Heap stores (-distance, id, neuron) to avoid comparing neurons
        heap = []
        
        self._query_knn_recursive(self.root, point, k, heap)
        
        # Extract neurons from heap and sort by distance (ascending)
        results = sorted([(abs(dist), neuron) for dist, _, neuron in heap], 
                        key=lambda x: x[0])
        return [neuron for _, neuron in results]
    
    def _query_knn_recursive(self, node: OctreeNode, point: Vector3D, 
                            k: int, heap: List[Tuple[float, str, Neuron]]):
        """
        Recursively search for k nearest neighbors.
        
        Args:
            node: Current node
            point: Query point
            k: Number of neighbors to find
            heap: Max heap of (-distance, id, neuron) tuples
        """
        # Check neurons in this node
        for neuron in node.neurons:
            distance = neuron.position.distance(point)
            
            if len(heap) < k:
                # Heap not full, add neuron
                heapq.heappush(heap, (-distance, str(neuron.id), neuron))
            elif distance < abs(heap[0][0]):
                # This neuron is closer than the farthest in heap
                heapq.heapreplace(heap, (-distance, str(neuron.id), neuron))
        
        # If this is a leaf, we're done with this branch
        if node.is_leaf():
            return
        
        # Calculate distances to each child and sort by distance
        child_distances = []
        for i, child in enumerate(node.children):
            # Distance from point to closest point in child's bounds
            closest_x = max(child.center.x - child.half_size,
                          min(point.x, child.center.x + child.half_size))
            closest_y = max(child.center.y - child.half_size,
                          min(point.y, child.center.y + child.half_size))
            closest_z = max(child.center.z - child.half_size,
                          min(point.z, child.center.z + child.half_size))
            
            closest_point = Vector3D(closest_x, closest_y, closest_z)
            dist = point.distance(closest_point)
            child_distances.append((dist, i))
        
        # Sort children by distance
        child_distances.sort()
        
        # Search children in order of proximity
        for dist, i in child_distances:
            # Prune if this child is farther than our kth nearest neighbor
            if len(heap) >= k and dist > abs(heap[0][0]):
                continue
            
            self._query_knn_recursive(node.children[i], point, k, heap)
    
    def clear(self):
        """Remove all neurons from the octree."""
        self.root.neurons = []
        self.root.children = None
        self.neuron_count = 0
    
    def get_neuron_count(self) -> int:
        """Get total number of neurons in the octree."""
        return self.neuron_count
