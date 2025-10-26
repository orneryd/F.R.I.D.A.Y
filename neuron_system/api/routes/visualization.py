"""
Visualization endpoints for exporting neuron network data.

Provides endpoints for exporting neuron positions, synapse connections,
and cluster information in formats suitable for 3D rendering (e.g., Three.js).
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from uuid import UUID
import math

from neuron_system.api.models import (
    VisualizationNeuronsResponse,
    VisualizationNeuronResponse,
    VisualizationSynapsesResponse,
    VisualizationSynapseResponse,
    VisualizationClustersResponse,
    VisualizationClusterResponse,
    ClusterBoundary,
    Vector3DModel
)
from neuron_system.core.vector3d import Vector3D
from pydantic import BaseModel

router = APIRouter()


# ============================================================================
# Three.js Compatible Models
# ============================================================================

class ThreeJSVector3(BaseModel):
    """Three.js compatible Vector3 format"""
    x: float
    y: float
    z: float


class ThreeJSColor(BaseModel):
    """Three.js compatible Color format"""
    r: float  # 0.0 to 1.0
    g: float  # 0.0 to 1.0
    b: float  # 0.0 to 1.0


class ThreeJSNeuron(BaseModel):
    """Three.js compatible neuron representation"""
    id: str
    position: ThreeJSVector3
    color: ThreeJSColor
    size: float
    type: str
    activation: float
    metadata: dict


class ThreeJSSynapse(BaseModel):
    """Three.js compatible synapse representation (line)"""
    id: str
    start: ThreeJSVector3  # Source neuron position
    end: ThreeJSVector3    # Target neuron position
    color: ThreeJSColor
    thickness: float
    weight: float
    type: str


class ThreeJSCluster(BaseModel):
    """Three.js compatible cluster representation (sphere)"""
    id: str
    name: str
    center: ThreeJSVector3
    radius: float
    color: ThreeJSColor
    neuronCount: int
    metadata: dict


class ThreeJSScene(BaseModel):
    """Complete Three.js scene data"""
    neurons: List[ThreeJSNeuron]
    synapses: List[ThreeJSSynapse]
    clusters: List[ThreeJSCluster]
    bounds: dict
    metadata: dict


def _get_app_state():
    """Get application state from app context"""
    from neuron_system.api.app import app_state
    return app_state


def _get_neuron_color(neuron_type: str, activation_level: float) -> str:
    """
    Calculate color for neuron based on type and activation.
    
    Args:
        neuron_type: Type of neuron
        activation_level: Activation level (0.0 to 1.0)
        
    Returns:
        Hex color code
    """
    # Base colors by type
    type_colors = {
        "knowledge": (100, 150, 255),  # Blue
        "tool": (255, 150, 100),       # Orange
        "memory": (150, 255, 100),     # Green
        "sensor": (255, 255, 100),     # Yellow
        "decision": (255, 100, 255),   # Magenta
    }
    
    base_color = type_colors.get(neuron_type, (200, 200, 200))  # Gray default
    
    # Brighten based on activation
    factor = 0.5 + (activation_level * 0.5)  # 0.5 to 1.0
    r = min(255, int(base_color[0] * factor))
    g = min(255, int(base_color[1] * factor))
    b = min(255, int(base_color[2] * factor))
    
    return f"#{r:02x}{g:02x}{b:02x}"


def _get_neuron_size(neuron, connection_count: int) -> float:
    """
    Calculate size for neuron based on connections and importance.
    
    Args:
        neuron: Neuron instance
        connection_count: Number of connections
        
    Returns:
        Relative size (1.0 = normal)
    """
    # Base size
    base_size = 1.0
    
    # Scale by connection count (more connections = larger)
    connection_factor = 1.0 + (connection_count * 0.05)
    connection_factor = min(connection_factor, 3.0)  # Cap at 3x
    
    # Scale by activation
    activation_factor = 0.8 + (neuron.activation_level * 0.4)  # 0.8 to 1.2
    
    return base_size * connection_factor * activation_factor


def _get_synapse_color(weight: float, synapse_type: str) -> str:
    """
    Calculate color for synapse based on weight and type.
    
    Args:
        weight: Synapse weight (-1.0 to 1.0)
        synapse_type: Type of synapse
        
    Returns:
        Hex color code
    """
    if weight >= 0:
        # Positive weights: green gradient
        intensity = int(weight * 255)
        return f"#{0:02x}{intensity:02x}{0:02x}"
    else:
        # Negative weights: red gradient
        intensity = int(abs(weight) * 255)
        return f"#{intensity:02x}{0:02x}{0:02x}"


def _get_synapse_thickness(weight: float) -> float:
    """
    Calculate thickness for synapse based on weight.
    
    Args:
        weight: Synapse weight (-1.0 to 1.0)
        
    Returns:
        Line thickness (0.1 to 2.0)
    """
    return 0.1 + (abs(weight) * 1.9)


def _hex_to_threejs_color(hex_color: str) -> ThreeJSColor:
    """
    Convert hex color to Three.js color format (0.0 to 1.0).
    
    Args:
        hex_color: Hex color string (e.g., "#ff8800")
        
    Returns:
        ThreeJSColor with normalized RGB values
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Parse RGB values
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    
    return ThreeJSColor(r=r, g=g, b=b)


def _get_cluster_color(cluster_index: int) -> ThreeJSColor:
    """
    Get a distinct color for a cluster.
    
    Args:
        cluster_index: Index of the cluster
        
    Returns:
        ThreeJSColor
    """
    # Predefined distinct colors
    colors = [
        (0.2, 0.6, 1.0),   # Blue
        (1.0, 0.6, 0.2),   # Orange
        (0.6, 1.0, 0.2),   # Green
        (1.0, 0.2, 0.6),   # Pink
        (0.6, 0.2, 1.0),   # Purple
        (1.0, 1.0, 0.2),   # Yellow
        (0.2, 1.0, 1.0),   # Cyan
        (1.0, 0.2, 0.2),   # Red
    ]
    
    color = colors[cluster_index % len(colors)]
    return ThreeJSColor(r=color[0], g=color[1], b=color[2])


def _calculate_cluster_boundary(neurons: List) -> ClusterBoundary:
    """
    Calculate bounding sphere for a cluster of neurons.
    
    Args:
        neurons: List of neuron instances
        
    Returns:
        ClusterBoundary with center and radius
    """
    if not neurons:
        return ClusterBoundary(
            center=Vector3DModel(x=0, y=0, z=0),
            radius=0,
            neuron_count=0
        )
    
    # Calculate centroid
    sum_x = sum(n.position.x for n in neurons)
    sum_y = sum(n.position.y for n in neurons)
    sum_z = sum(n.position.z for n in neurons)
    count = len(neurons)
    
    center = Vector3D(sum_x / count, sum_y / count, sum_z / count)
    
    # Calculate radius (max distance from center)
    max_distance = 0
    for neuron in neurons:
        distance = center.distance(neuron.position)
        max_distance = max(max_distance, distance)
    
    return ClusterBoundary(
        center=Vector3DModel(x=center.x, y=center.y, z=center.z),
        radius=max_distance,
        neuron_count=count
    )


@router.get(
    "/visualization/neurons",
    response_model=VisualizationNeuronsResponse,
    summary="Export neuron positions and metadata",
    description="Export neuron data for 3D visualization with positions, colors, and sizes"
)
async def get_visualization_neurons(
    neuron_type: Optional[str] = Query(None, description="Filter by neuron type"),
    min_activation: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum activation level"),
    max_activation: Optional[float] = Query(None, ge=0.0, le=1.0, description="Maximum activation level"),
    limit: Optional[int] = Query(None, ge=1, le=100000, description="Maximum number of neurons to return")
):
    """
    Export neuron positions and metadata for visualization.
    
    Returns neuron data in a format suitable for 3D rendering engines like Three.js.
    Includes position, type, activation level, and visual properties (color, size).
    
    Requirements: 6.1, 6.2, 6.4, 6.5
    """
    app_state = _get_app_state()
    
    if not app_state.graph:
        raise HTTPException(status_code=503, detail="Neuron graph not initialized")
    
    # Get all neurons
    neurons = list(app_state.graph.neurons.values())
    
    # Apply filters
    if neuron_type:
        neurons = [n for n in neurons if n.neuron_type.value == neuron_type]
    
    if min_activation is not None:
        neurons = [n for n in neurons if n.activation_level >= min_activation]
    
    if max_activation is not None:
        neurons = [n for n in neurons if n.activation_level <= max_activation]
    
    # Apply limit
    if limit:
        neurons = neurons[:limit]
    
    # Calculate bounds
    if neurons:
        min_x = min(n.position.x for n in neurons if n.position)
        max_x = max(n.position.x for n in neurons if n.position)
        min_y = min(n.position.y for n in neurons if n.position)
        max_y = max(n.position.y for n in neurons if n.position)
        min_z = min(n.position.z for n in neurons if n.position)
        max_z = max(n.position.z for n in neurons if n.position)
        
        bounds = {
            "min": Vector3DModel(x=min_x, y=min_y, z=min_z),
            "max": Vector3DModel(x=max_x, y=max_y, z=max_z)
        }
    else:
        bounds = {
            "min": Vector3DModel(x=0, y=0, z=0),
            "max": Vector3DModel(x=0, y=0, z=0)
        }
    
    # Build response
    visualization_neurons = []
    for neuron in neurons:
        if not neuron.position:
            continue
        
        # Count connections
        outgoing = len(app_state.graph.get_outgoing_synapses(neuron.id))
        incoming = len(app_state.graph.get_incoming_synapses(neuron.id))
        connection_count = outgoing + incoming
        
        # Calculate visual properties
        color = _get_neuron_color(neuron.neuron_type.value, neuron.activation_level)
        size = _get_neuron_size(neuron, connection_count)
        
        # Generate label
        label = f"{neuron.neuron_type.value}_{str(neuron.id)[:8]}"
        
        visualization_neurons.append(VisualizationNeuronResponse(
            id=str(neuron.id),
            neuron_type=neuron.neuron_type.value,
            position=Vector3DModel(
                x=neuron.position.x,
                y=neuron.position.y,
                z=neuron.position.z
            ),
            activation_level=neuron.activation_level,
            metadata=neuron.metadata,
            color=color,
            size=size,
            label=label
        ))
    
    return VisualizationNeuronsResponse(
        neurons=visualization_neurons,
        count=len(visualization_neurons),
        bounds=bounds
    )


@router.get(
    "/visualization/synapses",
    response_model=VisualizationSynapsesResponse,
    summary="Export synapse connections",
    description="Export synapse data for 3D visualization with weights and visual properties"
)
async def get_visualization_synapses(
    source_neuron_id: Optional[str] = Query(None, description="Filter by source neuron ID"),
    target_neuron_id: Optional[str] = Query(None, description="Filter by target neuron ID"),
    min_weight: Optional[float] = Query(None, ge=-1.0, le=1.0, description="Minimum synapse weight"),
    max_weight: Optional[float] = Query(None, ge=-1.0, le=1.0, description="Maximum synapse weight"),
    synapse_type: Optional[str] = Query(None, description="Filter by synapse type"),
    limit: Optional[int] = Query(None, ge=1, le=100000, description="Maximum number of synapses to return")
):
    """
    Export synapse connections for visualization.
    
    Returns synapse data in a format suitable for 3D rendering engines.
    Includes source/target neurons, weights, and visual properties (color, thickness).
    
    Requirements: 6.1, 6.2, 6.4, 6.5
    """
    app_state = _get_app_state()
    
    if not app_state.graph:
        raise HTTPException(status_code=503, detail="Neuron graph not initialized")
    
    # Get all synapses
    synapses = list(app_state.graph.synapses.values())
    
    # Apply filters
    if source_neuron_id:
        try:
            source_uuid = UUID(source_neuron_id)
            synapses = [s for s in synapses if s.source_neuron_id == source_uuid]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid source_neuron_id format")
    
    if target_neuron_id:
        try:
            target_uuid = UUID(target_neuron_id)
            synapses = [s for s in synapses if s.target_neuron_id == target_uuid]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid target_neuron_id format")
    
    if min_weight is not None:
        synapses = [s for s in synapses if s.weight >= min_weight]
    
    if max_weight is not None:
        synapses = [s for s in synapses if s.weight <= max_weight]
    
    if synapse_type:
        synapses = [s for s in synapses if s.synapse_type == synapse_type]
    
    # Apply limit
    if limit:
        synapses = synapses[:limit]
    
    # Build response
    visualization_synapses = []
    for synapse in synapses:
        # Calculate visual properties
        color = _get_synapse_color(synapse.weight, synapse.synapse_type)
        thickness = _get_synapse_thickness(synapse.weight)
        
        visualization_synapses.append(VisualizationSynapseResponse(
            id=str(synapse.id),
            source_neuron_id=str(synapse.source_neuron_id),
            target_neuron_id=str(synapse.target_neuron_id),
            weight=synapse.weight,
            synapse_type=synapse.synapse_type,
            color=color,
            thickness=thickness
        ))
    
    return VisualizationSynapsesResponse(
        synapses=visualization_synapses,
        count=len(visualization_synapses)
    )


@router.get(
    "/visualization/clusters",
    response_model=VisualizationClustersResponse,
    summary="Export neuron clusters",
    description="Export cluster information with boundaries for visualization"
)
async def get_visualization_clusters(
    cluster_name: Optional[str] = Query(None, description="Filter by cluster name (partial match)")
):
    """
    Export neuron clusters for visualization.
    
    Returns cluster data including member neurons and calculated boundaries.
    Useful for visualizing tool clusters and semantic groupings.
    
    Requirements: 6.1, 6.2, 6.4, 6.5
    """
    app_state = _get_app_state()
    
    if not app_state.graph:
        raise HTTPException(status_code=503, detail="Neuron graph not initialized")
    
    # Get all clusters
    clusters = list(app_state.graph.tool_clusters.values())
    
    # Apply name filter
    if cluster_name:
        clusters = [c for c in clusters if cluster_name.lower() in c.name.lower()]
    
    # Build response
    visualization_clusters = []
    for cluster in clusters:
        # Get cluster neurons
        cluster_neurons = []
        for neuron_id in cluster.tool_neurons:
            neuron = app_state.graph.get_neuron(neuron_id)
            if neuron and neuron.position:
                cluster_neurons.append(neuron)
        
        # Calculate boundary
        boundary = _calculate_cluster_boundary(cluster_neurons)
        
        visualization_clusters.append(VisualizationClusterResponse(
            cluster_id=str(cluster.id),
            name=cluster.name,
            neurons=[str(nid) for nid in cluster.tool_neurons],
            boundary=boundary,
            metadata=cluster.metadata
        ))
    
    return VisualizationClustersResponse(
        clusters=visualization_clusters,
        count=len(visualization_clusters)
    )



@router.get(
    "/visualization/threejs",
    response_model=ThreeJSScene,
    summary="Export complete scene in Three.js format",
    description="Export all visualization data in Three.js compatible format for direct rendering"
)
async def get_threejs_scene(
    neuron_type: Optional[str] = Query(None, description="Filter neurons by type"),
    min_activation: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum activation level"),
    include_synapses: bool = Query(True, description="Include synapse connections"),
    include_clusters: bool = Query(True, description="Include cluster boundaries"),
    max_neurons: Optional[int] = Query(10000, ge=1, le=100000, description="Maximum neurons to export"),
    max_synapses: Optional[int] = Query(50000, ge=1, le=100000, description="Maximum synapses to export")
):
    """
    Export complete scene data in Three.js compatible format.
    
    Returns a complete scene with neurons (as spheres), synapses (as lines),
    and clusters (as bounding spheres) ready for Three.js rendering.
    
    All positions are in 3D coordinates, colors are normalized RGB (0.0-1.0),
    and the format matches Three.js conventions for easy integration.
    
    Requirements: 6.3
    """
    app_state = _get_app_state()
    
    if not app_state.graph:
        raise HTTPException(status_code=503, detail="Neuron graph not initialized")
    
    # Get neurons
    neurons = list(app_state.graph.neurons.values())
    
    # Apply filters
    if neuron_type:
        neurons = [n for n in neurons if n.neuron_type.value == neuron_type]
    
    if min_activation is not None:
        neurons = [n for n in neurons if n.activation_level >= min_activation]
    
    # Apply limit
    if max_neurons:
        neurons = neurons[:max_neurons]
    
    # Build Three.js neurons
    threejs_neurons = []
    neuron_positions = {}  # Cache for synapse rendering
    
    for neuron in neurons:
        if not neuron.position:
            continue
        
        # Count connections
        outgoing = len(app_state.graph.get_outgoing_synapses(neuron.id))
        incoming = len(app_state.graph.get_incoming_synapses(neuron.id))
        connection_count = outgoing + incoming
        
        # Calculate visual properties
        hex_color = _get_neuron_color(neuron.neuron_type.value, neuron.activation_level)
        color = _hex_to_threejs_color(hex_color)
        size = _get_neuron_size(neuron, connection_count)
        
        # Store position for synapse rendering
        neuron_positions[neuron.id] = neuron.position
        
        threejs_neurons.append(ThreeJSNeuron(
            id=str(neuron.id),
            position=ThreeJSVector3(
                x=neuron.position.x,
                y=neuron.position.y,
                z=neuron.position.z
            ),
            color=color,
            size=size,
            type=neuron.neuron_type.value,
            activation=neuron.activation_level,
            metadata=neuron.metadata
        ))
    
    # Build Three.js synapses
    threejs_synapses = []
    
    if include_synapses:
        synapses = list(app_state.graph.synapses.values())
        
        # Filter to only synapses between visible neurons
        visible_neuron_ids = set(neuron_positions.keys())
        synapses = [
            s for s in synapses
            if s.source_neuron_id in visible_neuron_ids
            and s.target_neuron_id in visible_neuron_ids
        ]
        
        # Apply limit
        if max_synapses:
            synapses = synapses[:max_synapses]
        
        for synapse in synapses:
            source_pos = neuron_positions.get(synapse.source_neuron_id)
            target_pos = neuron_positions.get(synapse.target_neuron_id)
            
            if not source_pos or not target_pos:
                continue
            
            # Calculate visual properties
            hex_color = _get_synapse_color(synapse.weight, synapse.synapse_type)
            color = _hex_to_threejs_color(hex_color)
            thickness = _get_synapse_thickness(synapse.weight)
            
            threejs_synapses.append(ThreeJSSynapse(
                id=str(synapse.id),
                start=ThreeJSVector3(
                    x=source_pos.x,
                    y=source_pos.y,
                    z=source_pos.z
                ),
                end=ThreeJSVector3(
                    x=target_pos.x,
                    y=target_pos.y,
                    z=target_pos.z
                ),
                color=color,
                thickness=thickness,
                weight=synapse.weight,
                type=synapse.synapse_type
            ))
    
    # Build Three.js clusters
    threejs_clusters = []
    
    if include_clusters:
        clusters = list(app_state.graph.tool_clusters.values())
        
        for idx, cluster in enumerate(clusters):
            # Get cluster neurons
            cluster_neurons = []
            for neuron_id in cluster.tool_neurons:
                if neuron_id in neuron_positions:
                    neuron = app_state.graph.get_neuron(neuron_id)
                    if neuron:
                        cluster_neurons.append(neuron)
            
            if not cluster_neurons:
                continue
            
            # Calculate boundary
            boundary = _calculate_cluster_boundary(cluster_neurons)
            color = _get_cluster_color(idx)
            
            threejs_clusters.append(ThreeJSCluster(
                id=str(cluster.id),
                name=cluster.name,
                center=ThreeJSVector3(
                    x=boundary.center.x,
                    y=boundary.center.y,
                    z=boundary.center.z
                ),
                radius=boundary.radius,
                color=color,
                neuronCount=boundary.neuron_count,
                metadata=cluster.metadata
            ))
    
    # Calculate scene bounds
    if threejs_neurons:
        positions = [n.position for n in threejs_neurons]
        min_x = min(p.x for p in positions)
        max_x = max(p.x for p in positions)
        min_y = min(p.y for p in positions)
        max_y = max(p.y for p in positions)
        min_z = min(p.z for p in positions)
        max_z = max(p.z for p in positions)
        
        bounds = {
            "min": {"x": min_x, "y": min_y, "z": min_z},
            "max": {"x": max_x, "y": max_y, "z": max_z},
            "center": {
                "x": (min_x + max_x) / 2,
                "y": (min_y + max_y) / 2,
                "z": (min_z + max_z) / 2
            },
            "size": {
                "x": max_x - min_x,
                "y": max_y - min_y,
                "z": max_z - min_z
            }
        }
    else:
        bounds = {
            "min": {"x": 0, "y": 0, "z": 0},
            "max": {"x": 0, "y": 0, "z": 0},
            "center": {"x": 0, "y": 0, "z": 0},
            "size": {"x": 0, "y": 0, "z": 0}
        }
    
    # Build metadata
    metadata = {
        "neuronCount": len(threejs_neurons),
        "synapseCount": len(threejs_synapses),
        "clusterCount": len(threejs_clusters),
        "format": "threejs",
        "version": "1.0.0"
    }
    
    return ThreeJSScene(
        neurons=threejs_neurons,
        synapses=threejs_synapses,
        clusters=threejs_clusters,
        bounds=bounds,
        metadata=metadata
    )
