"""
Query Engine for finding relevant neurons and executing queries.

This module provides the QueryEngine class that handles knowledge queries,
spatial searches, and activation propagation through the neuron network.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from uuid import UUID
import numpy as np

from neuron_system.core.graph import NeuronGraph
from neuron_system.core.neuron import Neuron, NeuronType
from neuron_system.core.vector3d import Vector3D
from neuron_system.spatial.spatial_index import SpatialIndex
from neuron_system.engines.compression import CompressionEngine

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class ActivatedNeuron:
    """
    Represents a neuron with its activation level after query processing.
    
    Attributes:
        neuron: The neuron instance
        activation: Activation level (0.0 to 1.0)
        distance: Distance from query point (for spatial queries)
        similarity: Cosine similarity to query vector (for semantic queries)
        metadata: Additional metadata about activation
    """
    neuron: Neuron
    activation: float
    distance: Optional[float] = None
    similarity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare by activation level for sorting."""
        return self.activation < other.activation


class QueryEngine:
    """
    Engine for executing queries against the neuron network.
    
    Handles semantic queries using vector similarity, spatial queries,
    and activation propagation through synapses.
    
    Attributes:
        graph: The neuron graph to query
        compression_engine: Engine for compressing query text
        spatial_index: Spatial index for 3D queries
    """
    
    def __init__(
        self,
        neuron_graph: NeuronGraph,
        compression_engine: Optional[CompressionEngine] = None
    ):
        """
        Initialize QueryEngine.
        
        Args:
            neuron_graph: The neuron graph to query
            compression_engine: Optional compression engine (creates default if None)
        """
        self.graph = neuron_graph
        self.spatial_index = neuron_graph.spatial_index
        
        # Initialize or use provided compression engine
        if compression_engine is None:
            self.compression_engine = CompressionEngine()
        else:
            self.compression_engine = compression_engine
        
        # Query cache for performance
        self._query_cache: Dict[str, Tuple[List[ActivatedNeuron], float]] = {}
        self._cache_enabled = True
        self._cache_max_size = 100
        self._cache_ttl_seconds = 300  # 5 minutes
        
        # Performance tracking
        self._query_count = 0
        self._total_query_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        propagation_depth: int = 3,
        activation_threshold: float = 0.1,
        use_cache: bool = True
    ) -> List[ActivatedNeuron]:
        """
        Execute a semantic query to find relevant neurons.
        
        Process:
        1. Compress query text to vector
        2. Find nearest neurons in 3D space using spatial index
        3. Calculate initial activation using cosine similarity
        4. Propagate activation through synapses
        5. Return top-k activated neurons sorted by activation score
        
        Args:
            query_text: Text query to search for
            top_k: Number of top results to return
            propagation_depth: How many hops to propagate activation
            activation_threshold: Minimum activation level to include
            use_cache: Whether to use query cache
        
        Returns:
            List of ActivatedNeuron objects sorted by activation (highest first)
        
        Requirements: 5.1, 5.4, 15.1
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"{query_text}:{top_k}:{propagation_depth}:{activation_threshold}"
        if use_cache and self._cache_enabled:
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self._cache_hits += 1
                logger.debug(f"Cache hit for query: {query_text[:50]}")
                return cached_result
            self._cache_misses += 1
        
        try:
            # Step 1: Compress query to vector
            logger.debug(f"Compressing query: {query_text[:50]}")
            query_vector, compression_meta = self.compression_engine.compress(
                query_text,
                normalize=True
            )
            
            if not compression_meta.get("success", False):
                logger.warning(f"Query compression failed: {compression_meta.get('error')}")
                return []
            
            # Step 2: Find initial candidate neurons
            # Use spatial search to find neurons with vectors
            # Optimized: Reduce candidates for faster inference
            candidates = self._find_candidate_neurons(query_vector, top_k * 2)  # Reduced from 3x to 2x
            
            if not candidates:
                logger.warning("No candidate neurons found for query")
                return []
            
            # Step 3: Calculate initial activation using cosine similarity
            activated_neurons = self._calculate_initial_activation(
                candidates,
                query_vector,
                activation_threshold
            )
            
            if not activated_neurons:
                logger.warning("No neurons exceeded activation threshold")
                return []
            
            # Step 4: Propagate activation through synapses
            # This will be implemented in activation.py (subtask 5.2)
            # For now, we'll import and use it
            from neuron_system.engines.activation import propagate_activation
            
            activated_neurons = propagate_activation(
                self.graph,
                activated_neurons,
                depth=propagation_depth,
                threshold=activation_threshold
            )
            
            # Step 5: Sort and return top-k
            activated_neurons.sort(reverse=True)  # Sort by activation (highest first)
            result = activated_neurons[:top_k]
            
            # Update performance tracking
            elapsed_time = time.time() - start_time
            self._query_count += 1
            self._total_query_time += elapsed_time
            
            logger.info(
                f"Query completed in {elapsed_time*1000:.2f}ms, "
                f"found {len(result)} neurons"
            )
            
            # Cache result
            if use_cache and self._cache_enabled:
                self._add_to_cache(cache_key, result, elapsed_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}", exc_info=True)
            return []
    
    def _find_candidate_neurons(
        self,
        query_vector: np.ndarray,
        max_candidates: int
    ) -> List[Neuron]:
        """
        Find candidate neurons for initial activation.
        
        Returns all neurons with valid vectors for accurate similarity matching.
        
        Args:
            query_vector: Query embedding vector
            max_candidates: Maximum number of candidates to return (ignored for now)
        
        Returns:
            List of candidate neurons
        """
        # Get ALL neurons with vectors for accurate similarity matching
        # (We have only ~700 neurons, so this is fast enough)
        candidates = []
        for neuron in self.graph.neurons.values():
            if neuron.vector is not None and len(neuron.vector) > 0:
                candidates.append(neuron)
        
        logger.debug(f"Found {len(candidates)} candidate neurons with vectors")
        return candidates
    
    def _calculate_initial_activation(
        self,
        neurons: List[Neuron],
        query_vector: np.ndarray,
        threshold: float
    ) -> List[ActivatedNeuron]:
        """
        Calculate initial activation levels using cosine similarity.
        
        Args:
            neurons: List of candidate neurons
            query_vector: Query embedding vector
            threshold: Minimum activation threshold
        
        Returns:
            List of ActivatedNeuron objects with activation levels
        """
        activated = []
        
        for neuron in neurons:
            if neuron.vector is None or len(neuron.vector) == 0:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_vector, neuron.vector)
            
            # Convert similarity [-1, 1] to activation [0, 1]
            activation = (similarity + 1.0) / 2.0
            
            # Apply threshold
            if activation >= threshold:
                activated_neuron = ActivatedNeuron(
                    neuron=neuron,
                    activation=activation,
                    similarity=similarity,
                    metadata={
                        "initial_activation": activation,
                        "source": "query_similarity"
                    }
                )
                activated.append(activated_neuron)
        
        return activated
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Cosine similarity in range [-1, 1]
        """
        # Ensure vectors are the same length
        if len(vec1) != len(vec2):
            logger.warning(
                f"Vector length mismatch: {len(vec1)} vs {len(vec2)}, "
                "padding shorter vector"
            )
            max_len = max(len(vec1), len(vec2))
            if len(vec1) < max_len:
                vec1 = np.pad(vec1, (0, max_len - len(vec1)))
            if len(vec2) < max_len:
                vec2 = np.pad(vec2, (0, max_len - len(vec2)))
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Clamp to [-1, 1] to handle floating point errors
        return np.clip(similarity, -1.0, 1.0)
    
    def _get_from_cache(self, key: str) -> Optional[List[ActivatedNeuron]]:
        """
        Get query result from cache if available and not expired.
        
        Args:
            key: Cache key
        
        Returns:
            Cached result or None if not found/expired
        """
        if key not in self._query_cache:
            return None
        
        result, timestamp = self._query_cache[key]
        
        # Check if expired
        if time.time() - timestamp > self._cache_ttl_seconds:
            del self._query_cache[key]
            return None
        
        return result
    
    def _add_to_cache(
        self,
        key: str,
        result: List[ActivatedNeuron],
        timestamp: float = None
    ):
        """
        Add query result to cache.
        
        Args:
            key: Cache key
            result: Query result to cache
            timestamp: Optional timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Evict oldest entry if cache is full
        if len(self._query_cache) >= self._cache_max_size:
            oldest_key = min(
                self._query_cache.keys(),
                key=lambda k: self._query_cache[k][1]
            )
            del self._query_cache[oldest_key]
        
        self._query_cache[key] = (result, timestamp)
    
    def clear_cache(self):
        """Clear the query cache."""
        self._query_cache.clear()
        logger.info("Query cache cleared")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the query engine.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_time = (
            self._total_query_time / self._query_count
            if self._query_count > 0
            else 0.0
        )
        
        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0
            else 0.0
        )
        
        return {
            "total_queries": self._query_count,
            "total_time_seconds": self._total_query_time,
            "average_time_ms": avg_time * 1000,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._query_cache),
            "cache_enabled": self._cache_enabled
        }

    def spatial_query(
        self,
        center: Vector3D,
        radius: float,
        neuron_type_filter: Optional[str] = None,
        top_k: Optional[int] = None,
        activation_threshold: float = 0.0
    ) -> List[ActivatedNeuron]:
        """
        Execute a spatial query to find neurons in a 3D region.
        
        Finds neurons within a spherical region and optionally filters
        by neuron type.
        
        Args:
            center: Center point of the search region
            radius: Search radius
            neuron_type_filter: Optional neuron type to filter by (e.g., "knowledge", "tool")
            top_k: Optional limit on number of results
            activation_threshold: Minimum activation level (based on distance)
        
        Returns:
            List of ActivatedNeuron objects sorted by distance (nearest first)
        
        Requirements: 5.5, 15.1
        """
        start_time = time.time()
        
        try:
            # Query spatial index for neurons in radius
            logger.debug(f"Spatial query: center={center}, radius={radius}")
            neurons = self.spatial_index.query_radius(center, radius)
            
            if not neurons:
                logger.debug("No neurons found in spatial region")
                return []
            
            # Filter by neuron type if specified
            if neuron_type_filter:
                neurons = [
                    n for n in neurons
                    if n.neuron_type.value == neuron_type_filter
                ]
                logger.debug(
                    f"Filtered to {len(neurons)} neurons of type {neuron_type_filter}"
                )
            
            # Calculate activation based on distance (closer = higher activation)
            activated_neurons = []
            for neuron in neurons:
                distance = center.distance(neuron.position)
                
                # Convert distance to activation (inverse relationship)
                # activation = 1.0 at center, decreases to 0.0 at radius
                if radius > 0:
                    activation = max(0.0, 1.0 - (distance / radius))
                else:
                    activation = 1.0 if distance == 0 else 0.0
                
                # Apply threshold
                if activation >= activation_threshold:
                    activated_neuron = ActivatedNeuron(
                        neuron=neuron,
                        activation=activation,
                        distance=distance,
                        metadata={
                            "query_type": "spatial",
                            "center": center.to_dict(),
                            "radius": radius,
                            "distance": distance
                        }
                    )
                    activated_neurons.append(activated_neuron)
            
            # Sort by distance (nearest first)
            activated_neurons.sort(key=lambda x: x.distance)
            
            # Limit to top_k if specified
            if top_k is not None:
                activated_neurons = activated_neurons[:top_k]
            
            elapsed_time = time.time() - start_time
            logger.info(
                f"Spatial query completed in {elapsed_time*1000:.2f}ms, "
                f"found {len(activated_neurons)} neurons"
            )
            
            return activated_neurons
            
        except Exception as e:
            logger.error(f"Spatial query failed: {str(e)}", exc_info=True)
            return []
    
    def spatial_query_with_propagation(
        self,
        center: Vector3D,
        radius: float,
        neuron_type_filter: Optional[str] = None,
        propagation_depth: int = 2,
        top_k: int = 10
    ) -> List[ActivatedNeuron]:
        """
        Execute a spatial query and propagate activation.
        
        Combines spatial search with activation propagation to find
        neurons that are spatially close or connected to the region.
        
        Args:
            center: Center point of the search region
            radius: Search radius
            neuron_type_filter: Optional neuron type to filter by
            propagation_depth: How many hops to propagate activation
            top_k: Number of top results to return
        
        Returns:
            List of ActivatedNeuron objects sorted by activation
        """
        # Get initial neurons from spatial query
        initial_neurons = self.spatial_query(
            center=center,
            radius=radius,
            neuron_type_filter=neuron_type_filter
        )
        
        if not initial_neurons:
            return []
        
        # Propagate activation
        from neuron_system.engines.activation import propagate_activation
        
        activated_neurons = propagate_activation(
            self.graph,
            initial_neurons,
            depth=propagation_depth,
            threshold=0.1
        )
        
        # Sort and return top-k
        activated_neurons.sort(reverse=True)
        return activated_neurons[:top_k]
    
    def query_by_neuron_type(
        self,
        neuron_type: str,
        query_text: Optional[str] = None,
        top_k: int = 10
    ) -> List[ActivatedNeuron]:
        """
        Query neurons filtered by type.
        
        Args:
            neuron_type: Type of neurons to query (e.g., "knowledge", "tool")
            query_text: Optional text query for semantic filtering
            top_k: Number of results to return
        
        Returns:
            List of ActivatedNeuron objects of the specified type
        """
        # Get all neurons of the specified type
        neurons = [
            n for n in self.graph.neurons.values()
            if n.neuron_type.value == neuron_type
        ]
        
        if not neurons:
            logger.warning(f"No neurons found of type: {neuron_type}")
            return []
        
        # If query text provided, use semantic similarity
        if query_text:
            query_vector, _ = self.compression_engine.compress(query_text, normalize=True)
            activated_neurons = self._calculate_initial_activation(
                neurons,
                query_vector,
                threshold=0.0
            )
            activated_neurons.sort(reverse=True)
            return activated_neurons[:top_k]
        
        # Otherwise, return neurons with default activation
        activated_neurons = []
        for neuron in neurons[:top_k]:
            activated_neuron = ActivatedNeuron(
                neuron=neuron,
                activation=0.5,  # Default activation
                metadata={"query_type": "type_filter", "neuron_type": neuron_type}
            )
            activated_neurons.append(activated_neuron)
        
        return activated_neurons
    
    def query_with_timeout(
        self,
        query_text: str,
        timeout_seconds: float = 5.0,
        **kwargs
    ) -> List[ActivatedNeuron]:
        """
        Execute a query with a timeout mechanism.
        
        Args:
            query_text: Text query to search for
            timeout_seconds: Maximum time to allow for query
            **kwargs: Additional arguments passed to query()
        
        Returns:
            List of ActivatedNeuron objects, or empty list if timeout
        """
        import threading
        
        result = []
        exception = None
        
        def query_thread():
            nonlocal result, exception
            try:
                result = self.query(query_text, **kwargs)
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=query_thread)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            logger.warning(f"Query timed out after {timeout_seconds}s")
            return []
        
        if exception:
            logger.error(f"Query failed: {str(exception)}")
            return []
        
        return result
    
    def query_region(
        self,
        min_bound: Vector3D,
        max_bound: Vector3D,
        neuron_type_filter: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[ActivatedNeuron]:
        """
        Query neurons within a rectangular 3D region.
        
        Args:
            min_bound: Minimum corner of the region
            max_bound: Maximum corner of the region
            neuron_type_filter: Optional neuron type to filter by
            top_k: Optional limit on number of results
        
        Returns:
            List of ActivatedNeuron objects in the region
        """
        # Query spatial index for region
        neurons = self.spatial_index.query_region(min_bound, max_bound)
        
        # Filter by type if specified
        if neuron_type_filter:
            neurons = [
                n for n in neurons
                if n.neuron_type.value == neuron_type_filter
            ]
        
        # Calculate center of region for distance calculations
        center = Vector3D(
            (min_bound.x + max_bound.x) / 2,
            (min_bound.y + max_bound.y) / 2,
            (min_bound.z + max_bound.z) / 2
        )
        
        # Create activated neurons
        activated_neurons = []
        for neuron in neurons:
            distance = center.distance(neuron.position)
            
            activated_neuron = ActivatedNeuron(
                neuron=neuron,
                activation=0.5,  # Default activation for region queries
                distance=distance,
                metadata={
                    "query_type": "region",
                    "min_bound": min_bound.to_dict(),
                    "max_bound": max_bound.to_dict()
                }
            )
            activated_neurons.append(activated_neuron)
        
        # Sort by distance from center
        activated_neurons.sort(key=lambda x: x.distance)
        
        # Limit to top_k if specified
        if top_k is not None:
            activated_neurons = activated_neurons[:top_k]
        
        return activated_neurons
    
    def get_neuron_neighbors(
        self,
        neuron_id: UUID,
        max_neighbors: int = 10,
        include_incoming: bool = False
    ) -> List[ActivatedNeuron]:
        """
        Get connected neighbors of a neuron.
        
        Args:
            neuron_id: UUID of the neuron
            max_neighbors: Maximum number of neighbors to return
            include_incoming: Whether to include incoming connections
        
        Returns:
            List of ActivatedNeuron objects representing neighbors
        """
        neuron = self.graph.get_neuron(neuron_id)
        if not neuron:
            logger.warning(f"Neuron not found: {neuron_id}")
            return []
        
        neighbors = []
        
        # Get outgoing neighbors
        for synapse, target_neuron in self.graph.get_neighbors(neuron_id):
            activation = abs(synapse.weight)  # Use synapse weight as activation
            
            activated_neighbor = ActivatedNeuron(
                neuron=target_neuron,
                activation=activation,
                metadata={
                    "query_type": "neighbors",
                    "source_neuron_id": str(neuron_id),
                    "synapse_id": str(synapse.id),
                    "synapse_weight": synapse.weight,
                    "direction": "outgoing"
                }
            )
            neighbors.append(activated_neighbor)
        
        # Get incoming neighbors if requested
        if include_incoming:
            incoming_synapses = self.graph.get_incoming_synapses(neuron_id)
            for synapse in incoming_synapses:
                source_neuron = self.graph.get_neuron(synapse.source_neuron_id)
                if source_neuron:
                    activation = abs(synapse.weight)
                    
                    activated_neighbor = ActivatedNeuron(
                        neuron=source_neuron,
                        activation=activation,
                        metadata={
                            "query_type": "neighbors",
                            "target_neuron_id": str(neuron_id),
                            "synapse_id": str(synapse.id),
                            "synapse_weight": synapse.weight,
                            "direction": "incoming"
                        }
                    )
                    neighbors.append(activated_neighbor)
        
        # Sort by activation (synapse weight)
        neighbors.sort(reverse=True)
        
        return neighbors[:max_neighbors]
    
    def batch_query(
        self,
        query_texts: List[str],
        top_k: int = 10,
        **kwargs
    ) -> List[List[ActivatedNeuron]]:
        """
        Execute multiple queries in batch.
        
        Args:
            query_texts: List of text queries
            top_k: Number of results per query
            **kwargs: Additional arguments passed to query()
        
        Returns:
            List of result lists, one per query
        """
        results = []
        
        for query_text in query_texts:
            result = self.query(query_text, top_k=top_k, **kwargs)
            results.append(result)
        
        return results
    
    def enable_cache(self, enabled: bool = True):
        """
        Enable or disable query caching.
        
        Args:
            enabled: Whether to enable caching
        """
        self._cache_enabled = enabled
        logger.info(f"Query cache {'enabled' if enabled else 'disabled'}")
    
    def set_cache_config(
        self,
        max_size: Optional[int] = None,
        ttl_seconds: Optional[float] = None
    ):
        """
        Configure cache settings.
        
        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time-to-live for cached results
        """
        if max_size is not None:
            self._cache_max_size = max_size
            logger.info(f"Cache max size set to {max_size}")
        
        if ttl_seconds is not None:
            self._cache_ttl_seconds = ttl_seconds
            logger.info(f"Cache TTL set to {ttl_seconds}s")
    
    def reset_stats(self):
        """Reset performance tracking statistics."""
        self._query_count = 0
        self._total_query_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Query engine statistics reset")
    
    def execute_tool_neurons(
        self,
        activated_neurons: List[ActivatedNeuron],
        execution_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect and execute activated Tool Neurons.
        
        Identifies Tool Neurons that exceed the activation threshold,
        executes them with inputs from connected neurons, and collects results.
        
        Args:
            activated_neurons: List of activated neurons from query
            execution_threshold: Minimum activation to trigger tool execution
            
        Returns:
            Dictionary with tool execution results and statistics
            
        Requirements: 10.2
        """
        tool_results = {
            "executed_tools": [],
            "failed_tools": [],
            "total_executed": 0,
            "total_failed": 0
        }
        
        # Build activation map for input extraction
        activation_map = {
            an.neuron.id: an.activation
            for an in activated_neurons
        }
        
        # Find and execute tool neurons
        for activated_neuron in activated_neurons:
            neuron = activated_neuron.neuron
            
            # Check if this is a tool neuron
            if neuron.neuron_type != NeuronType.TOOL:
                continue
            
            # Check if activation exceeds threshold
            if activated_neuron.activation < execution_threshold:
                logger.debug(
                    f"Tool neuron {neuron.id} activation {activated_neuron.activation:.3f} "
                    f"below threshold {execution_threshold}"
                )
                continue
            
            # Execute the tool
            try:
                logger.info(
                    f"Executing tool neuron {neuron.id} with activation "
                    f"{activated_neuron.activation:.3f}"
                )
                
                # Extract inputs from connected neurons
                inputs = neuron.extract_inputs_from_synapses(self.graph, activation_map)
                
                # Execute the tool
                result = neuron.execute(inputs)
                
                # Propagate results to output synapses
                propagated_to = neuron.propagate_results_to_outputs(self.graph, result)
                
                # Record successful execution
                tool_results["executed_tools"].append({
                    "neuron_id": str(neuron.id),
                    "activation": activated_neuron.activation,
                    "inputs": inputs,
                    "result": result,
                    "propagated_to": [str(nid) for nid in propagated_to],
                    "execution_count": neuron.execution_count,
                    "average_execution_time": neuron.average_execution_time
                })
                tool_results["total_executed"] += 1
                
                logger.info(
                    f"Tool neuron {neuron.id} executed successfully, "
                    f"propagated to {len(propagated_to)} neurons"
                )
                
            except Exception as e:
                logger.error(
                    f"Tool neuron {neuron.id} execution failed: {str(e)}",
                    exc_info=True
                )
                
                # Handle the error
                error_info = neuron.handle_execution_error(self.graph, e)
                
                # Record failed execution
                tool_results["failed_tools"].append({
                    "neuron_id": str(neuron.id),
                    "activation": activated_neuron.activation,
                    "error": error_info
                })
                tool_results["total_failed"] += 1
        
        return tool_results
    
    def query_with_tool_execution(
        self,
        query_text: str,
        top_k: int = 10,
        propagation_depth: int = 3,
        activation_threshold: float = 0.1,
        tool_execution_threshold: float = 0.5,
        include_tool_results: bool = True
    ) -> Tuple[List[ActivatedNeuron], Dict[str, Any]]:
        """
        Execute a query and automatically execute activated tool neurons.
        
        This is a convenience method that combines query() and execute_tool_neurons().
        
        Args:
            query_text: Text query to search for
            top_k: Number of top results to return
            propagation_depth: How many hops to propagate activation
            activation_threshold: Minimum activation level to include
            tool_execution_threshold: Minimum activation to trigger tool execution
            include_tool_results: Whether to include tool execution results
            
        Returns:
            Tuple of (activated_neurons, tool_results)
            
        Requirements: 10.2
        """
        # Execute the query
        activated_neurons = self.query(
            query_text=query_text,
            top_k=top_k,
            propagation_depth=propagation_depth,
            activation_threshold=activation_threshold
        )
        
        # Execute tool neurons if requested
        tool_results = {}
        if include_tool_results:
            tool_results = self.execute_tool_neurons(
                activated_neurons,
                execution_threshold=tool_execution_threshold
            )
        
        return activated_neurons, tool_results
    
    def aggregate_tool_results(
        self,
        tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate and format tool execution results for response.
        
        Collects all tool execution results and formats them for
        inclusion in the query response.
        
        Args:
            tool_results: Raw tool execution results from execute_tool_neurons()
            
        Returns:
            Aggregated and formatted tool results
            
        Requirements: 10.2
        """
        aggregated = {
            "summary": {
                "total_tools_executed": tool_results.get("total_executed", 0),
                "total_tools_failed": tool_results.get("total_failed", 0),
                "success_rate": 0.0
            },
            "successful_executions": [],
            "failed_executions": []
        }
        
        # Calculate success rate
        total = aggregated["summary"]["total_tools_executed"] + aggregated["summary"]["total_tools_failed"]
        if total > 0:
            aggregated["summary"]["success_rate"] = (
                aggregated["summary"]["total_tools_executed"] / total
            )
        
        # Format successful executions
        for tool_exec in tool_results.get("executed_tools", []):
            aggregated["successful_executions"].append({
                "tool_id": tool_exec["neuron_id"],
                "activation": tool_exec["activation"],
                "result": tool_exec["result"],
                "execution_time_ms": tool_exec.get("average_execution_time", 0),
                "propagated_to_count": len(tool_exec.get("propagated_to", []))
            })
        
        # Format failed executions
        for tool_fail in tool_results.get("failed_tools", []):
            aggregated["failed_executions"].append({
                "tool_id": tool_fail["neuron_id"],
                "activation": tool_fail["activation"],
                "error": tool_fail["error"]["error_message"]
            })
        
        return aggregated
