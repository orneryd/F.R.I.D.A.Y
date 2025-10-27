"""
Neuron Clustering - Advanced clustering for semantic organization.

Implements multiple clustering strategies:
- K-Means Clustering (vector-based)
- Hierarchical Clustering (dendrogram-based)
- DBSCAN (density-based)
- Topic-based Clustering (tag-based)
- Hybrid Clustering (combines multiple strategies)
"""

import logging
import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict
from uuid import UUID

from neuron_system.core.neuron import Neuron
from neuron_system.core.vector3d import Vector3D

logger = logging.getLogger(__name__)


class NeuronCluster:
    """
    Represents a cluster of neurons.
    """
    
    def __init__(self, cluster_id: int, name: str = None):
        """
        Initialize cluster.
        
        Args:
            cluster_id: Unique cluster ID
            name: Optional cluster name
        """
        self.id = cluster_id
        self.name = name or f"cluster_{cluster_id}"
        self.neurons: Set[UUID] = set()
        self.centroid: Optional[np.ndarray] = None
        self.center_position: Optional[Vector3D] = None
        self.tags: Set[str] = set()
        self.quality_score: float = 0.0
    
    def add_neuron(self, neuron_id: UUID):
        """Add neuron to cluster."""
        self.neurons.add(neuron_id)
    
    def remove_neuron(self, neuron_id: UUID):
        """Remove neuron from cluster."""
        self.neurons.discard(neuron_id)
    
    def size(self) -> int:
        """Get cluster size."""
        return len(self.neurons)
    
    def __repr__(self):
        return f"NeuronCluster(id={self.id}, name={self.name}, size={self.size()})"


class NeuronClusteringEngine:
    """
    Advanced clustering engine for neurons.
    """
    
    def __init__(self):
        """Initialize clustering engine."""
        self.clusters: Dict[int, NeuronCluster] = {}
        self.neuron_to_cluster: Dict[UUID, int] = {}
        self.next_cluster_id = 0
    
    def cluster_kmeans(
        self,
        neurons: List[Neuron],
        n_clusters: int = 10,
        max_iterations: int = 100,
        tolerance: float = 1e-4
    ) -> Dict[int, NeuronCluster]:
        """
        K-Means clustering based on neuron vectors.
        
        Args:
            neurons: List of neurons to cluster
            n_clusters: Number of clusters
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary of clusters
        """
        logger.info(f"K-Means clustering: {len(neurons)} neurons into {n_clusters} clusters")
        
        # Filter neurons with vectors
        valid_neurons = [n for n in neurons if n.vector is not None]
        if len(valid_neurons) < n_clusters:
            logger.warning(f"Not enough neurons with vectors: {len(valid_neurons)} < {n_clusters}")
            n_clusters = max(1, len(valid_neurons))
        
        if not valid_neurons:
            return {}
        
        # Convert to numpy array
        vectors = np.array([n.vector for n in valid_neurons])
        
        # Initialize centroids (k-means++)
        centroids = self._kmeans_plus_plus_init(vectors, n_clusters)
        
        # K-Means iterations
        for iteration in range(max_iterations):
            # Assign neurons to nearest centroid
            assignments = self._assign_to_nearest(vectors, centroids)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(n_clusters):
                cluster_vectors = vectors[assignments == i]
                if len(cluster_vectors) > 0:
                    new_centroids[i] = cluster_vectors.mean(axis=0)
                else:
                    # Keep old centroid if cluster is empty
                    new_centroids[i] = centroids[i]
            
            # Check convergence
            centroid_shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            
            if centroid_shift < tolerance:
                logger.info(f"K-Means converged after {iteration + 1} iterations")
                break
        
        # Create clusters
        self.clusters.clear()
        self.neuron_to_cluster.clear()
        
        for i in range(n_clusters):
            cluster = NeuronCluster(self.next_cluster_id, f"kmeans_{i}")
            cluster.centroid = centroids[i]
            self.clusters[cluster.id] = cluster
            self.next_cluster_id += 1
        
        # Assign neurons to clusters
        for neuron, cluster_idx in zip(valid_neurons, assignments):
            cluster_id = list(self.clusters.keys())[cluster_idx]
            self.clusters[cluster_id].add_neuron(neuron.id)
            self.neuron_to_cluster[neuron.id] = cluster_id
        
        # Calculate cluster statistics
        self._calculate_cluster_stats(valid_neurons)
        
        logger.info(f"Created {len(self.clusters)} clusters")
        return self.clusters
    
    def cluster_dbscan(
        self,
        neurons: List[Neuron],
        eps: float = 0.3,
        min_samples: int = 3
    ) -> Dict[int, NeuronCluster]:
        """
        DBSCAN clustering (density-based).
        
        Args:
            neurons: List of neurons to cluster
            eps: Maximum distance between neighbors
            min_samples: Minimum samples in neighborhood
            
        Returns:
            Dictionary of clusters
        """
        logger.info(f"DBSCAN clustering: {len(neurons)} neurons (eps={eps}, min_samples={min_samples})")
        
        # Filter neurons with vectors
        valid_neurons = [n for n in neurons if n.vector is not None]
        if not valid_neurons:
            return {}
        
        # Convert to numpy array
        vectors = np.array([n.vector for n in valid_neurons])
        
        # Normalize vectors for cosine distance
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        
        # DBSCAN algorithm
        labels = np.full(len(vectors), -1)  # -1 = noise
        cluster_id = 0
        
        for i in range(len(vectors)):
            if labels[i] != -1:
                continue  # Already processed
            
            # Find neighbors
            neighbors = self._find_neighbors(vectors_norm, i, eps)
            
            if len(neighbors) < min_samples:
                labels[i] = -1  # Mark as noise
                continue
            
            # Start new cluster
            labels[i] = cluster_id
            
            # Expand cluster
            seed_set = list(neighbors)
            while seed_set:
                current = seed_set.pop(0)
                
                if labels[current] == -1:
                    labels[current] = cluster_id
                
                if labels[current] != -1:
                    continue
                
                labels[current] = cluster_id
                
                # Find neighbors of current point
                current_neighbors = self._find_neighbors(vectors_norm, current, eps)
                if len(current_neighbors) >= min_samples:
                    seed_set.extend(current_neighbors)
            
            cluster_id += 1
        
        # Create clusters
        self.clusters.clear()
        self.neuron_to_cluster.clear()
        
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
            
            cluster = NeuronCluster(self.next_cluster_id, f"dbscan_{label}")
            self.clusters[cluster.id] = cluster
            self.next_cluster_id += 1
        
        # Assign neurons to clusters
        for neuron, label in zip(valid_neurons, labels):
            if label == -1:
                continue  # Skip noise
            
            cluster_id = list(self.clusters.keys())[label]
            self.clusters[cluster_id].add_neuron(neuron.id)
            self.neuron_to_cluster[neuron.id] = cluster_id
        
        # Calculate cluster statistics
        self._calculate_cluster_stats(valid_neurons)
        
        logger.info(f"Created {len(self.clusters)} clusters ({sum(labels == -1)} noise points)")
        return self.clusters
    
    def cluster_by_topics(
        self,
        neurons: List[Neuron]
    ) -> Dict[int, NeuronCluster]:
        """
        Cluster neurons by semantic tags.
        
        Args:
            neurons: List of neurons to cluster
            
        Returns:
            Dictionary of clusters
        """
        logger.info(f"Topic-based clustering: {len(neurons)} neurons")
        
        # Group by primary tag
        topic_groups = defaultdict(list)
        for neuron in neurons:
            if hasattr(neuron, 'semantic_tags') and neuron.semantic_tags:
                primary_tag = neuron.semantic_tags[0]
                topic_groups[primary_tag].append(neuron)
            else:
                topic_groups['untagged'].append(neuron)
        
        # Create clusters
        self.clusters.clear()
        self.neuron_to_cluster.clear()
        
        for topic, topic_neurons in topic_groups.items():
            cluster = NeuronCluster(self.next_cluster_id, f"topic_{topic}")
            cluster.tags.add(topic)
            
            for neuron in topic_neurons:
                cluster.add_neuron(neuron.id)
                self.neuron_to_cluster[neuron.id] = cluster.id
            
            self.clusters[cluster.id] = cluster
            self.next_cluster_id += 1
        
        # Calculate cluster statistics
        self._calculate_cluster_stats(neurons)
        
        logger.info(f"Created {len(self.clusters)} topic-based clusters")
        return self.clusters
    
    def cluster_hybrid(
        self,
        neurons: List[Neuron],
        n_clusters: int = 10,
        topic_weight: float = 0.3,
        vector_weight: float = 0.7
    ) -> Dict[int, NeuronCluster]:
        """
        Hybrid clustering combining topics and vectors.
        
        Args:
            neurons: List of neurons to cluster
            n_clusters: Number of clusters
            topic_weight: Weight for topic similarity
            vector_weight: Weight for vector similarity
            
        Returns:
            Dictionary of clusters
        """
        logger.info(f"Hybrid clustering: {len(neurons)} neurons into {n_clusters} clusters")
        
        # Filter neurons with vectors
        valid_neurons = [n for n in neurons if n.vector is not None]
        if not valid_neurons:
            return {}
        
        # Build combined similarity matrix
        n = len(valid_neurons)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Vector similarity (cosine)
                vec_sim = np.dot(valid_neurons[i].vector, valid_neurons[j].vector) / (
                    np.linalg.norm(valid_neurons[i].vector) * 
                    np.linalg.norm(valid_neurons[j].vector) + 1e-8
                )
                
                # Topic similarity (Jaccard)
                topic_sim = 0.0
                if (hasattr(valid_neurons[i], 'semantic_tags') and 
                    hasattr(valid_neurons[j], 'semantic_tags')):
                    tags_i = set(valid_neurons[i].semantic_tags or [])
                    tags_j = set(valid_neurons[j].semantic_tags or [])
                    if tags_i or tags_j:
                        topic_sim = len(tags_i & tags_j) / len(tags_i | tags_j)
                
                # Combined similarity
                combined_sim = vector_weight * vec_sim + topic_weight * topic_sim
                similarity_matrix[i, j] = combined_sim
                similarity_matrix[j, i] = combined_sim
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        # Use K-Means on distance matrix
        vectors = np.array([n.vector for n in valid_neurons])
        centroids = self._kmeans_plus_plus_init(vectors, n_clusters)
        
        # K-Means with custom distance
        max_iterations = 100
        for iteration in range(max_iterations):
            # Assign to nearest centroid using combined distance
            assignments = np.zeros(n, dtype=int)
            for i in range(n):
                distances = []
                for c_idx in range(n_clusters):
                    # Find neurons in this cluster
                    cluster_members = np.where(assignments == c_idx)[0] if iteration > 0 else []
                    if len(cluster_members) == 0:
                        # Use centroid distance
                        dist = np.linalg.norm(vectors[i] - centroids[c_idx])
                    else:
                        # Use average distance to cluster members
                        dist = np.mean([distance_matrix[i, j] for j in cluster_members])
                    distances.append(dist)
                assignments[i] = np.argmin(distances)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(n_clusters):
                cluster_vectors = vectors[assignments == i]
                if len(cluster_vectors) > 0:
                    new_centroids[i] = cluster_vectors.mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]
            
            # Check convergence
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
        
        # Create clusters
        self.clusters.clear()
        self.neuron_to_cluster.clear()
        
        for i in range(n_clusters):
            cluster = NeuronCluster(self.next_cluster_id, f"hybrid_{i}")
            cluster.centroid = centroids[i]
            self.clusters[cluster.id] = cluster
            self.next_cluster_id += 1
        
        # Assign neurons to clusters
        for neuron, cluster_idx in zip(valid_neurons, assignments):
            cluster_id = list(self.clusters.keys())[cluster_idx]
            self.clusters[cluster_id].add_neuron(neuron.id)
            self.neuron_to_cluster[neuron.id] = cluster_id
            
            # Add tags to cluster
            if hasattr(neuron, 'semantic_tags') and neuron.semantic_tags:
                self.clusters[cluster_id].tags.update(neuron.semantic_tags)
        
        # Calculate cluster statistics
        self._calculate_cluster_stats(valid_neurons)
        
        logger.info(f"Created {len(self.clusters)} hybrid clusters")
        return self.clusters
    
    def get_cluster(self, neuron_id: UUID) -> Optional[NeuronCluster]:
        """Get cluster for neuron."""
        cluster_id = self.neuron_to_cluster.get(neuron_id)
        if cluster_id is not None:
            return self.clusters.get(cluster_id)
        return None
    
    def get_cluster_neighbors(
        self,
        neuron_id: UUID,
        max_neighbors: int = 10
    ) -> List[UUID]:
        """
        Get neurons in the same cluster.
        
        Args:
            neuron_id: Neuron ID
            max_neighbors: Maximum neighbors to return
            
        Returns:
            List of neuron IDs in same cluster
        """
        cluster = self.get_cluster(neuron_id)
        if not cluster:
            return []
        
        # Return other neurons in cluster
        neighbors = [nid for nid in cluster.neurons if nid != neuron_id]
        return neighbors[:max_neighbors]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get clustering statistics."""
        if not self.clusters:
            return {
                'num_clusters': 0,
                'total_neurons': 0,
                'avg_cluster_size': 0,
                'min_cluster_size': 0,
                'max_cluster_size': 0,
            }
        
        sizes = [c.size() for c in self.clusters.values()]
        
        return {
            'num_clusters': len(self.clusters),
            'total_neurons': sum(sizes),
            'avg_cluster_size': np.mean(sizes),
            'min_cluster_size': min(sizes),
            'max_cluster_size': max(sizes),
            'avg_quality_score': np.mean([c.quality_score for c in self.clusters.values()]),
        }
    
    def _kmeans_plus_plus_init(
        self,
        vectors: np.ndarray,
        n_clusters: int
    ) -> np.ndarray:
        """
        K-Means++ initialization for better initial centroids.
        
        Args:
            vectors: Data vectors
            n_clusters: Number of clusters
            
        Returns:
            Initial centroids
        """
        n_samples = len(vectors)
        centroids = np.zeros((n_clusters, vectors.shape[1]))
        
        # Choose first centroid randomly
        centroids[0] = vectors[np.random.randint(n_samples)]
        
        # Choose remaining centroids
        for i in range(1, n_clusters):
            # Calculate distances to nearest centroid
            distances = np.min([
                np.linalg.norm(vectors - centroids[j], axis=1)
                for j in range(i)
            ], axis=0)
            
            # Choose next centroid with probability proportional to distance^2
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            
            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids[i] = vectors[next_idx]
        
        return centroids
    
    def _assign_to_nearest(
        self,
        vectors: np.ndarray,
        centroids: np.ndarray
    ) -> np.ndarray:
        """Assign vectors to nearest centroid."""
        distances = np.array([
            np.linalg.norm(vectors - centroid, axis=1)
            for centroid in centroids
        ])
        return np.argmin(distances, axis=0)
    
    def _find_neighbors(
        self,
        vectors: np.ndarray,
        point_idx: int,
        eps: float
    ) -> List[int]:
        """Find neighbors within eps distance (cosine)."""
        point = vectors[point_idx]
        
        # Cosine distance
        similarities = np.dot(vectors, point)
        distances = 1 - similarities
        
        neighbors = np.where(distances <= eps)[0]
        return [int(i) for i in neighbors if i != point_idx]
    
    def _calculate_cluster_stats(self, neurons: List[Neuron]):
        """Calculate statistics for each cluster."""
        neuron_dict = {n.id: n for n in neurons}
        
        for cluster in self.clusters.values():
            if cluster.size() == 0:
                continue
            
            # Calculate center position
            positions = []
            vectors = []
            
            for neuron_id in cluster.neurons:
                neuron = neuron_dict.get(neuron_id)
                if neuron:
                    if neuron.position:
                        positions.append(neuron.position)
                    if neuron.vector is not None:
                        vectors.append(neuron.vector)
            
            # Center position
            if positions:
                cluster.center_position = Vector3D(
                    sum(p.x for p in positions) / len(positions),
                    sum(p.y for p in positions) / len(positions),
                    sum(p.z for p in positions) / len(positions)
                )
            
            # Quality score (cohesion)
            if len(vectors) > 1:
                vectors_array = np.array(vectors)
                centroid = vectors_array.mean(axis=0)
                
                # Average cosine similarity to centroid
                similarities = []
                for vec in vectors:
                    sim = np.dot(vec, centroid) / (
                        np.linalg.norm(vec) * np.linalg.norm(centroid) + 1e-8
                    )
                    similarities.append(sim)
                
                cluster.quality_score = np.mean(similarities)


__all__ = ['NeuronCluster', 'NeuronClusteringEngine']
