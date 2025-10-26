"""
Main NeuronGraph class for managing the neuron network.
"""

from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID
from collections import defaultdict
from datetime import datetime
import numpy as np

from neuron_system.core.neuron import Neuron, NeuronTypeRegistry
from neuron_system.core.synapse import Synapse
from neuron_system.core.vector3d import Vector3D
from neuron_system.spatial.spatial_index import SpatialIndex
from neuron_system.spatial.positioning import NeuronPositioner
from neuron_system.utils.uuid_pool import UUIDPool


class NeuronGraph:
    """
    Main graph structure for managing neurons and synapses.
    
    Maintains the complete network state including spatial organization
    and connection topology.
    """
    
    def __init__(self, bounds: Tuple[Vector3D, Vector3D] = None):
        """
        Initialize neuron graph.
        
        Args:
            bounds: Tuple of (min_bound, max_bound) for 3D space
        """
        self.neurons: Dict[UUID, Neuron] = {}
        self.synapses: Dict[UUID, Synapse] = {}
        
        # Index for fast synapse lookups
        self._outgoing_synapses: Dict[UUID, List[UUID]] = defaultdict(list)
        self._incoming_synapses: Dict[UUID, List[UUID]] = defaultdict(list)
        
        # Tool clusters storage
        self.tool_clusters: Dict[UUID, 'ToolCluster'] = {}
        self._cluster_by_name: Dict[str, UUID] = {}
        
        # Spatial bounds
        if bounds is None:
            self.bounds = (Vector3D(-100, -100, -100), Vector3D(100, 100, 100))
        else:
            self.bounds = bounds
        
        # Spatial index for 3D queries
        self.spatial_index = SpatialIndex(self.bounds)
        self.positioner = NeuronPositioner(self.bounds)
        
        # UUID pool for fast neuron creation
        self.uuid_pool = UUIDPool(initial_size=10000, refill_threshold=1000)
        
        # Optional storage layer references
        self._neuron_store = None
        self._synapse_store = None
    
    def attach_storage(self, neuron_store=None, synapse_store=None):
        """
        Attach storage layer for automatic persistence.
        
        Args:
            neuron_store: NeuronStore instance
            synapse_store: SynapseStore instance
        """
        if neuron_store:
            self._neuron_store = neuron_store
        if synapse_store:
            self._synapse_store = synapse_store
    
    @property
    def neuron_store(self):
        """Get attached neuron store."""
        return self._neuron_store
    
    @property
    def synapse_store(self):
        """Get attached synapse store."""
        return self._synapse_store
    
    def create_neuron(
        self,
        neuron_type: str,
        position: Optional[Vector3D] = None,
        vector: Optional[Any] = None,
        auto_position: bool = True,
        positioning_strategy: str = "random",
        lazy_vector: bool = True,
        **kwargs
    ) -> Neuron:
        """
        Create a new neuron with fast, type-agnostic creation.
        
        This method achieves < 1ms creation time by:
        - Using pre-allocated UUIDs from pool
        - Supporting lazy vector generation (deferred until first query)
        - Automatic positioning based on existing neurons
        - Ensuring neurons stay within spatial bounds
        
        Args:
            neuron_type: Type of neuron to create (e.g., "knowledge", "tool")
            position: Optional 3D position (auto-calculated if None and auto_position=True)
            vector: Optional embedding vector (can be None for lazy generation)
            auto_position: Whether to automatically calculate position
            positioning_strategy: Strategy for auto-positioning ("random", "similar", "cluster", "grid")
            lazy_vector: If True, defer vector generation until first query
            **kwargs: Additional type-specific parameters
            
        Returns:
            Created Neuron instance
            
        Raises:
            ValueError: If neuron type is not registered
            
        Requirements: 1.2, 8.1, 14.1, 14.4, 14.5
        """
        # Step 1: Get pre-allocated UUID from pool (< 0.01ms)
        neuron_id = self.uuid_pool.acquire()
        
        # Step 2: Create neuron instance via Registry factory (< 0.1ms)
        neuron = NeuronTypeRegistry.create(neuron_type, **kwargs)
        
        # Step 3: Assign UUID and timestamps
        neuron.id = neuron_id
        neuron.created_at = datetime.utcnow()
        neuron.modified_at = neuron.created_at
        
        # Step 4: Handle vector (lazy or immediate)
        if not lazy_vector and vector is not None:
            neuron.vector = vector
        elif lazy_vector:
            # Vector will be generated on first query
            neuron.vector = None
        
        # Step 5: Calculate optimal 3D position (< 0.5ms)
        if position is not None:
            # Use provided position
            neuron.position = position
        elif auto_position:
            # Auto-calculate position based on strategy
            if positioning_strategy == "similar" and neuron.vector is not None:
                # Position near semantically similar neurons
                existing = list(self.neurons.values())
                neuron.position = self.positioner.position_near_similar(
                    neuron, existing, k=kwargs.get("k", 5), spread=kwargs.get("spread", 5.0)
                )
            elif positioning_strategy == "cluster":
                # Position in a cluster
                center = kwargs.get("center", Vector3D(0, 0, 0))
                radius = kwargs.get("radius", 10.0)
                neuron.position = self.positioner.position_in_cluster(center, radius)
            elif positioning_strategy == "grid":
                # Position on a grid
                index = len(self.neurons)
                grid_size = kwargs.get("grid_size", 10)
                neuron.position = self.positioner.position_grid(index, grid_size)
            else:
                # Default: random positioning
                neuron.position = self.positioner.position_random()
            
            # Ensure position is within bounds
            neuron.position = self._clamp_position(neuron.position)
        else:
            # No position assigned yet
            neuron.position = None
        
        # Step 6: Add to graph
        self.neurons[neuron.id] = neuron
        
        # Step 7: Update spatial index automatically (< 0.3ms)
        if neuron.position:
            self.spatial_index.insert(neuron)
        
        return neuron
    
    def add_neuron(self, neuron: Neuron) -> UUID:
        """
        Add an existing neuron instance to the graph.
        
        Args:
            neuron: Neuron instance to add
            
        Returns:
            UUID of the added neuron
            
        Raises:
            ValueError: If neuron ID already exists
        """
        if neuron.id in self.neurons:
            raise ValueError(f"Neuron with ID {neuron.id} already exists")
        
        # Ensure vector is a numpy array if it exists
        if neuron.vector is not None and not isinstance(neuron.vector, np.ndarray):
            neuron.vector = np.asarray(neuron.vector)
        
        self.neurons[neuron.id] = neuron
        
        # Auto-insert into spatial index if neuron has position
        if neuron.position:
            self.spatial_index.insert(neuron)
        
        return neuron.id
    
    def _clamp_position(self, position: Vector3D) -> Vector3D:
        """
        Clamp position to spatial bounds.
        
        Args:
            position: Position to clamp
            
        Returns:
            Clamped position within bounds
        """
        return Vector3D(
            max(self.bounds[0].x, min(position.x, self.bounds[1].x)),
            max(self.bounds[0].y, min(position.y, self.bounds[1].y)),
            max(self.bounds[0].z, min(position.z, self.bounds[1].z))
        )
    
    def remove_neuron(self, neuron_id: UUID, update_storage: bool = False) -> bool:
        """
        Remove a neuron and all its synapses from the graph.
        
        Implements cascade deletion:
        - Deletes all associated synapses automatically
        - Removes from spatial index
        - Optionally updates storage layer
        
        Args:
            neuron_id: UUID of neuron to remove
            update_storage: If True, also delete from persistent storage
            
        Returns:
            True if neuron was removed, False if not found
            
        Requirements: 8.2
        """
        if neuron_id not in self.neurons:
            return False
        
        # Remove all synapses connected to this neuron (cascade)
        synapses_to_remove = []
        
        # Outgoing synapses
        for synapse_id in self._outgoing_synapses.get(neuron_id, []):
            synapses_to_remove.append(synapse_id)
        
        # Incoming synapses
        for synapse_id in self._incoming_synapses.get(neuron_id, []):
            synapses_to_remove.append(synapse_id)
        
        # Remove synapses
        for synapse_id in synapses_to_remove:
            self.remove_synapse(synapse_id, update_storage=update_storage)
        
        # Remove from spatial index
        self.spatial_index.remove(neuron_id)
        
        # Remove neuron from graph
        del self.neurons[neuron_id]
        
        # Clean up indices
        if neuron_id in self._outgoing_synapses:
            del self._outgoing_synapses[neuron_id]
        if neuron_id in self._incoming_synapses:
            del self._incoming_synapses[neuron_id]
        
        # Update storage layer if requested
        if update_storage and hasattr(self, '_neuron_store'):
            try:
                self._neuron_store.delete(neuron_id)
            except Exception as e:
                # Log error but don't fail the operation
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to delete neuron {neuron_id} from storage: {e}"
                )
        
        return True
    
    def get_neuron(self, neuron_id: UUID) -> Optional[Neuron]:
        """
        Get a neuron by ID.
        
        Args:
            neuron_id: UUID of neuron
            
        Returns:
            Neuron instance or None if not found
        """
        return self.neurons.get(neuron_id)
    
    def add_synapse(self, synapse: Synapse) -> UUID:
        """
        Add a synapse to the graph.
        
        Args:
            synapse: Synapse instance to add
            
        Returns:
            UUID of the added synapse
            
        Raises:
            ValueError: If synapse ID already exists or neurons don't exist
        """
        if synapse.id in self.synapses:
            raise ValueError(f"Synapse with ID {synapse.id} already exists")
        
        # Validate that both neurons exist
        if synapse.source_neuron_id not in self.neurons:
            raise ValueError(f"Source neuron {synapse.source_neuron_id} does not exist")
        if synapse.target_neuron_id not in self.neurons:
            raise ValueError(f"Target neuron {synapse.target_neuron_id} does not exist")
        
        self.synapses[synapse.id] = synapse
        
        # Update indices
        self._outgoing_synapses[synapse.source_neuron_id].append(synapse.id)
        self._incoming_synapses[synapse.target_neuron_id].append(synapse.id)
        
        return synapse.id
    
    def remove_synapse(self, synapse_id: UUID, update_storage: bool = False) -> bool:
        """
        Remove a synapse from the graph.
        
        Args:
            synapse_id: UUID of synapse to remove
            update_storage: If True, also delete from persistent storage
            
        Returns:
            True if synapse was removed, False if not found
        """
        if synapse_id not in self.synapses:
            return False
        
        synapse = self.synapses[synapse_id]
        
        # Update indices
        if synapse.source_neuron_id in self._outgoing_synapses:
            self._outgoing_synapses[synapse.source_neuron_id].remove(synapse_id)
        if synapse.target_neuron_id in self._incoming_synapses:
            self._incoming_synapses[synapse.target_neuron_id].remove(synapse_id)
        
        # Remove synapse from graph
        del self.synapses[synapse_id]
        
        # Update storage layer if requested
        if update_storage and hasattr(self, '_synapse_store'):
            try:
                self._synapse_store.delete(synapse_id)
            except Exception as e:
                # Log error but don't fail the operation
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to delete synapse {synapse_id} from storage: {e}"
                )
        
        return True
    
    def get_synapse(self, synapse_id: UUID) -> Optional[Synapse]:
        """
        Get a synapse by ID.
        
        Args:
            synapse_id: UUID of synapse
            
        Returns:
            Synapse instance or None if not found
        """
        return self.synapses.get(synapse_id)
    
    def get_outgoing_synapses(self, neuron_id: UUID) -> List[Synapse]:
        """
        Get all outgoing synapses from a neuron.
        
        Args:
            neuron_id: UUID of source neuron
            
        Returns:
            List of outgoing synapses
        """
        synapse_ids = self._outgoing_synapses.get(neuron_id, [])
        return [self.synapses[sid] for sid in synapse_ids if sid in self.synapses]
    
    def get_incoming_synapses(self, neuron_id: UUID) -> List[Synapse]:
        """
        Get all incoming synapses to a neuron.
        
        Args:
            neuron_id: UUID of target neuron
            
        Returns:
            List of incoming synapses
        """
        synapse_ids = self._incoming_synapses.get(neuron_id, [])
        return [self.synapses[sid] for sid in synapse_ids if sid in self.synapses]
    
    def get_neighbors(self, neuron_id: UUID) -> List[Tuple[Synapse, Neuron]]:
        """
        Get all connected neurons via outgoing synapses.
        
        Args:
            neuron_id: UUID of source neuron
            
        Returns:
            List of (synapse, target_neuron) tuples
        """
        neighbors = []
        for synapse in self.get_outgoing_synapses(neuron_id):
            target_neuron = self.neurons.get(synapse.target_neuron_id)
            if target_neuron:
                neighbors.append((synapse, target_neuron))
        return neighbors
    
    def get_neuron_count(self) -> int:
        """Get total number of neurons in the graph."""
        return len(self.neurons)
    
    def get_synapse_count(self) -> int:
        """Get total number of synapses in the graph."""
        return len(self.synapses)
    
    def clear(self):
        """Remove all neurons and synapses from the graph."""
        self.neurons.clear()
        self.synapses.clear()
        self._outgoing_synapses.clear()
        self._incoming_synapses.clear()
        self.spatial_index.clear()
    
    # Spatial query methods
    
    def query_radius(self, center: Vector3D, radius: float) -> List[Neuron]:
        """
        Find all neurons within a given radius of a center point.
        
        Args:
            center: Center point for the query
            radius: Search radius
            
        Returns:
            List of neurons within the radius
        """
        return self.spatial_index.query_radius(center, radius)
    
    def query_knn(self, point: Vector3D, k: int) -> List[Neuron]:
        """
        Find k nearest neighbors to a point.
        
        Args:
            point: Query point
            k: Number of neighbors to find
            
        Returns:
            List of k nearest neurons
        """
        return self.spatial_index.query_knn(point, k)
    
    def query_region(self, min_bound: Vector3D, max_bound: Vector3D) -> List[Neuron]:
        """
        Find all neurons within a rectangular region.
        
        Args:
            min_bound: Minimum corner of the region
            max_bound: Maximum corner of the region
            
        Returns:
            List of neurons within the region
        """
        return self.spatial_index.query_region(min_bound, max_bound)
    
    def rebalance_spatial_index(self):
        """
        Rebalance the spatial index when density exceeds threshold.
        
        This rebuilds the octree structure for optimal performance.
        """
        if self.spatial_index.needs_rebalancing():
            neurons_list = list(self.neurons.values())
            self.spatial_index.rebuild(neurons_list)
    
    def position_neuron(self, neuron: Neuron, strategy: str = "random", **kwargs) -> Vector3D:
        """
        Automatically position a neuron in 3D space.
        
        Args:
            neuron: Neuron to position
            strategy: Positioning strategy ("random", "similar", "cluster", "grid")
            **kwargs: Additional arguments for positioning strategy
            
        Returns:
            Calculated position
        """
        if strategy == "random":
            return self.positioner.position_random()
        elif strategy == "similar":
            existing = list(self.neurons.values())
            k = kwargs.get("k", 5)
            spread = kwargs.get("spread", 5.0)
            return self.positioner.position_near_similar(neuron, existing, k, spread)
        elif strategy == "cluster":
            center = kwargs.get("center", Vector3D(0, 0, 0))
            radius = kwargs.get("radius", 10.0)
            return self.positioner.position_in_cluster(center, radius)
        elif strategy == "grid":
            index = kwargs.get("index", 0)
            grid_size = kwargs.get("grid_size", 10)
            return self.positioner.position_grid(index, grid_size)
        else:
            raise ValueError(f"Unknown positioning strategy: {strategy}")
    
    def get_spatial_density(self, center: Vector3D, radius: float) -> float:
        """
        Calculate neuron density in a region.
        
        Args:
            center: Center of the region
            radius: Radius of the spherical region
            
        Returns:
            Density (neurons per cubic unit)
        """
        return self.spatial_index.get_density(center, radius)
    
    # Tool Cluster Management Methods
    
    def add_cluster(self, cluster: 'ToolCluster') -> UUID:
        """
        Add a tool cluster to the graph.
        
        Args:
            cluster: ToolCluster instance to add
            
        Returns:
            UUID of the added cluster
            
        Raises:
            ValueError: If cluster ID or name already exists
            
        Requirements: 11.5
        """
        from neuron_system.tools.tool_cluster import ToolCluster
        
        if not isinstance(cluster, ToolCluster):
            raise ValueError("cluster must be a ToolCluster instance")
        
        if cluster.id in self.tool_clusters:
            raise ValueError(f"Cluster with ID {cluster.id} already exists")
        
        if cluster.name in self._cluster_by_name:
            raise ValueError(f"Cluster with name '{cluster.name}' already exists")
        
        # Validate that all tool neurons exist in the graph
        for tool_id in cluster.tool_neurons:
            if tool_id not in self.neurons:
                raise ValueError(f"Tool neuron {tool_id} not found in graph")
        
        self.tool_clusters[cluster.id] = cluster
        self._cluster_by_name[cluster.name] = cluster.id
        
        return cluster.id
    
    def remove_cluster(self, cluster_id: UUID) -> bool:
        """
        Remove a tool cluster from the graph.
        
        Args:
            cluster_id: UUID of cluster to remove
            
        Returns:
            True if cluster was removed, False if not found
            
        Requirements: 11.5
        """
        if cluster_id not in self.tool_clusters:
            return False
        
        cluster = self.tool_clusters[cluster_id]
        
        # Remove from name index
        if cluster.name in self._cluster_by_name:
            del self._cluster_by_name[cluster.name]
        
        # Remove cluster
        del self.tool_clusters[cluster_id]
        
        return True
    
    def get_cluster(self, cluster_id: UUID) -> Optional['ToolCluster']:
        """
        Get a tool cluster by ID.
        
        Args:
            cluster_id: UUID of cluster
            
        Returns:
            ToolCluster instance or None if not found
        """
        return self.tool_clusters.get(cluster_id)
    
    def get_cluster_by_name(self, name: str) -> Optional['ToolCluster']:
        """
        Get a tool cluster by name.
        
        Args:
            name: Name of the cluster
            
        Returns:
            ToolCluster instance or None if not found
            
        Requirements: 11.5
        """
        cluster_id = self._cluster_by_name.get(name)
        if cluster_id:
            return self.tool_clusters.get(cluster_id)
        return None
    
    def query_clusters_by_capability(self, capability: str) -> List['ToolCluster']:
        """
        Query clusters by capability keyword.
        
        Searches cluster names and metadata for matching capabilities.
        
        Args:
            capability: Capability keyword to search for
            
        Returns:
            List of matching tool clusters
            
        Requirements: 11.5
        """
        matching_clusters = []
        capability_lower = capability.lower()
        
        for cluster in self.tool_clusters.values():
            # Check name
            if capability_lower in cluster.name.lower():
                matching_clusters.append(cluster)
                continue
            
            # Check metadata
            cluster_capabilities = cluster.metadata.get("capabilities", [])
            if isinstance(cluster_capabilities, list):
                if any(capability_lower in cap.lower() for cap in cluster_capabilities):
                    matching_clusters.append(cluster)
                    continue
            
            # Check description
            description = cluster.metadata.get("description", "")
            if capability_lower in description.lower():
                matching_clusters.append(cluster)
        
        return matching_clusters
    
    def execute_cluster(
        self,
        cluster_id: UUID,
        inputs: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool cluster as a callable unit.
        
        Args:
            cluster_id: UUID of the cluster to execute
            inputs: Input parameters for the cluster
            context: Additional execution context
            
        Returns:
            Dictionary with execution results
            
        Raises:
            ValueError: If cluster not found
            RuntimeError: If execution fails
            
        Requirements: 11.5
        """
        from neuron_system.tools.tool_executor import ToolClusterExecutor
        
        cluster = self.get_cluster(cluster_id)
        if cluster is None:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        executor = ToolClusterExecutor(self)
        return executor.execute_cluster(cluster, inputs, context)
    
    def execute_cluster_by_name(
        self,
        name: str,
        inputs: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool cluster by name.
        
        Args:
            name: Name of the cluster to execute
            inputs: Input parameters for the cluster
            context: Additional execution context
            
        Returns:
            Dictionary with execution results
            
        Raises:
            ValueError: If cluster not found
            RuntimeError: If execution fails
            
        Requirements: 11.5
        """
        cluster = self.get_cluster_by_name(name)
        if cluster is None:
            raise ValueError(f"Cluster '{name}' not found")
        
        return self.execute_cluster(cluster.id, inputs, context)
    
    def get_cluster_count(self) -> int:
        """Get total number of tool clusters in the graph."""
        return len(self.tool_clusters)
    
    def list_clusters(self) -> List['ToolCluster']:
        """
        Get all tool clusters in the graph.
        
        Returns:
            List of all tool clusters
        """
        return list(self.tool_clusters.values())
    
    # Batch Operations for High-Throughput
    
    def batch_create_neurons(
        self,
        neuron_specs: List[Dict[str, Any]],
        auto_position: bool = True,
        positioning_strategy: str = "random",
        lazy_vector: bool = True,
        use_transaction: bool = True,
        persist: bool = False
    ) -> List[Neuron]:
        """
        Create multiple neurons in a single batch operation.
        
        Optimized for high-throughput creation (10,000+ neurons/second):
        - Uses transactions for atomicity
        - Optimizes spatial index updates for batches
        - Pre-allocates UUIDs from pool
        - Supports lazy vector generation
        
        Args:
            neuron_specs: List of neuron specifications, each containing:
                - neuron_type: Type of neuron (required)
                - position: Optional position
                - vector: Optional vector
                - **kwargs: Type-specific parameters
            auto_position: Whether to automatically calculate positions
            positioning_strategy: Strategy for auto-positioning
            lazy_vector: If True, defer vector generation
            use_transaction: If True, use atomic transaction (all or nothing)
            persist: If True, persist to storage layer
            
        Returns:
            List of created Neuron instances
            
        Raises:
            ValueError: If any neuron spec is invalid
            RuntimeError: If transaction fails and use_transaction=True
            
        Requirements: 8.3, 14.2
        """
        if not neuron_specs:
            return []
        
        created_neurons = []
        
        try:
            # Create all neurons
            for spec in neuron_specs:
                neuron_type = spec.get("neuron_type")
                if not neuron_type:
                    raise ValueError("neuron_type is required in neuron spec")
                
                # Extract parameters
                position = spec.get("position")
                vector = spec.get("vector")
                kwargs = {k: v for k, v in spec.items() 
                         if k not in ["neuron_type", "position", "vector"]}
                
                # Create neuron using fast creation method
                neuron = self.create_neuron(
                    neuron_type=neuron_type,
                    position=position,
                    vector=vector,
                    auto_position=auto_position,
                    positioning_strategy=positioning_strategy,
                    lazy_vector=lazy_vector,
                    **kwargs
                )
                
                created_neurons.append(neuron)
            
            # Optimize spatial index updates for batch
            if len(created_neurons) > 100:
                # Rebalance spatial index after large batch
                self.rebalance_spatial_index()
            
            # Persist to storage if requested
            if persist and self._neuron_store:
                self._neuron_store.batch_create(created_neurons)
            
            return created_neurons
        
        except Exception as e:
            if use_transaction:
                # Rollback: remove all created neurons
                for neuron in created_neurons:
                    if neuron.id in self.neurons:
                        self.remove_neuron(neuron.id, update_storage=False)
                raise RuntimeError(f"Batch neuron creation failed: {e}")
            else:
                # Partial success: return what was created
                return created_neurons
    
    def batch_add_synapses(
        self,
        synapse_specs: List[Dict[str, Any]],
        use_transaction: bool = True,
        persist: bool = False
    ) -> List[Synapse]:
        """
        Create multiple synapses in a single batch operation.
        
        Optimized for high-throughput creation:
        - Uses transactions for atomicity
        - Validates all references before creation
        - Batch persistence to storage
        
        Args:
            synapse_specs: List of synapse specifications, each containing:
                - source_neuron_id: UUID of source neuron (required)
                - target_neuron_id: UUID of target neuron (required)
                - weight: Synapse weight (default: 0.5)
                - synapse_type: Type of synapse (default: "KNOWLEDGE")
                - **kwargs: Additional parameters
            use_transaction: If True, use atomic transaction (all or nothing)
            persist: If True, persist to storage layer
            
        Returns:
            List of created Synapse instances
            
        Raises:
            ValueError: If any synapse spec is invalid or neurons don't exist
            RuntimeError: If transaction fails and use_transaction=True
            
        Requirements: 8.3, 14.2
        """
        if not synapse_specs:
            return []
        
        created_synapses = []
        
        try:
            # Validate all neuron references first
            for spec in synapse_specs:
                source_id = spec.get("source_neuron_id")
                target_id = spec.get("target_neuron_id")
                
                if not source_id or not target_id:
                    raise ValueError("source_neuron_id and target_neuron_id are required")
                
                if source_id not in self.neurons:
                    raise ValueError(f"Source neuron {source_id} does not exist")
                if target_id not in self.neurons:
                    raise ValueError(f"Target neuron {target_id} does not exist")
            
            # Create all synapses
            for spec in synapse_specs:
                # Get pre-allocated UUID
                synapse_id = self.uuid_pool.acquire()
                
                # Create synapse
                synapse = Synapse()
                synapse.id = synapse_id
                synapse.source_neuron_id = UUID(spec["source_neuron_id"]) if isinstance(spec["source_neuron_id"], str) else spec["source_neuron_id"]
                synapse.target_neuron_id = UUID(spec["target_neuron_id"]) if isinstance(spec["target_neuron_id"], str) else spec["target_neuron_id"]
                synapse.weight = spec.get("weight", 0.5)
                synapse.synapse_type = spec.get("synapse_type", "KNOWLEDGE")
                synapse.usage_count = spec.get("usage_count", 0)
                synapse.metadata = spec.get("metadata", {})
                synapse.created_at = datetime.utcnow()
                synapse.modified_at = synapse.created_at
                synapse.last_traversed = None
                
                # Add to graph
                self.add_synapse(synapse)
                created_synapses.append(synapse)
            
            # Persist to storage if requested
            if persist and self._synapse_store:
                self._synapse_store.batch_create(created_synapses)
            
            return created_synapses
        
        except Exception as e:
            if use_transaction:
                # Rollback: remove all created synapses
                for synapse in created_synapses:
                    if synapse.id in self.synapses:
                        self.remove_synapse(synapse.id, update_storage=False)
                raise RuntimeError(f"Batch synapse creation failed: {e}")
            else:
                # Partial success: return what was created
                return created_synapses
    
    def batch_remove_neurons(
        self,
        neuron_ids: List[UUID],
        update_storage: bool = False
    ) -> int:
        """
        Remove multiple neurons in a batch operation.
        
        Args:
            neuron_ids: List of neuron UUIDs to remove
            update_storage: If True, also delete from persistent storage
            
        Returns:
            Number of neurons successfully removed
        """
        count = 0
        for neuron_id in neuron_ids:
            if self.remove_neuron(neuron_id, update_storage=update_storage):
                count += 1
        return count
    
    def batch_remove_synapses(
        self,
        synapse_ids: List[UUID],
        update_storage: bool = False
    ) -> int:
        """
        Remove multiple synapses in a batch operation.
        
        Args:
            synapse_ids: List of synapse UUIDs to remove
            update_storage: If True, also delete from persistent storage
            
        Returns:
            Number of synapses successfully removed
        """
        count = 0
        for synapse_id in synapse_ids:
            if self.remove_synapse(synapse_id, update_storage=update_storage):
                count += 1
        return count
