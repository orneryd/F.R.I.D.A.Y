"""
Brain layout algorithms for better 3D visualization
"""
import numpy as np
from typing import List, Tuple


def generate_spherical_position(index: int, total: int, radius: float = 50.0) -> Tuple[float, float, float]:
    """
    Generate position on a sphere using Fibonacci spiral
    Creates evenly distributed points on a sphere surface
    
    Args:
        index: Current neuron index
        total: Total number of neurons
        radius: Sphere radius
        
    Returns:
        (x, y, z) coordinates centered at origin
    """
    # Golden angle in radians
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    # Normalize index to [-1, 1]
    y = 1 - (index / float(total - 1)) * 2
    
    # Radius at y
    radius_at_y = np.sqrt(1 - y * y)
    
    # Angle
    theta = golden_angle * index
    
    # Calculate coordinates
    x = np.cos(theta) * radius_at_y
    z = np.sin(theta) * radius_at_y
    
    # Scale by radius
    return (
        float(x * radius),
        float(y * radius),
        float(z * radius)
    )


def generate_brain_shaped_position(index: int, total: int, scale: float = 40.0) -> Tuple[float, float, float]:
    """
    Generate position in a brain-like ellipsoid shape
    Wider in the middle, narrower at top and bottom
    
    Args:
        index: Current neuron index
        total: Total number of neurons
        scale: Overall size scale
        
    Returns:
        (x, y, z) coordinates centered at origin
    """
    # Use Fibonacci spiral as base
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    # Normalize index
    t = index / float(max(total - 1, 1))
    
    # Y coordinate (vertical) - slightly compressed
    y = (t - 0.5) * 2.0
    
    # Brain-like shape: wider in middle, narrower at ends
    # Use ellipsoid with different radii
    radius_xz = np.sqrt(1 - (y * 0.7) ** 2)  # Horizontal radius
    radius_y = 0.8  # Vertical compression
    
    # Angle for spiral
    theta = golden_angle * index
    
    # Add some randomness for organic look
    noise_x = np.random.uniform(-0.1, 0.1)
    noise_z = np.random.uniform(-0.1, 0.1)
    
    # Calculate coordinates
    x = (np.cos(theta) * radius_xz + noise_x) * scale
    z = (np.sin(theta) * radius_xz + noise_z) * scale
    y = y * radius_y * scale
    
    return (float(x), float(y), float(z))


def generate_clustered_position(
    index: int, 
    total: int, 
    cluster_id: int = 0, 
    num_clusters: int = 5,
    spread: float = 30.0
) -> Tuple[float, float, float]:
    """
    Generate position in clustered regions (for semantic grouping)
    
    Args:
        index: Current neuron index
        total: Total number of neurons
        cluster_id: Which cluster this neuron belongs to
        num_clusters: Total number of clusters
        spread: How spread out the clusters are
        
    Returns:
        (x, y, z) coordinates
    """
    # Generate cluster center positions on a sphere
    cluster_angle = (2 * np.pi * cluster_id) / num_clusters
    cluster_height = np.cos(np.pi * cluster_id / num_clusters)
    
    cluster_x = np.cos(cluster_angle) * np.sqrt(1 - cluster_height**2) * spread
    cluster_y = cluster_height * spread
    cluster_z = np.sin(cluster_angle) * np.sqrt(1 - cluster_height**2) * spread
    
    # Add random offset within cluster
    offset_radius = spread * 0.3
    offset_angle = np.random.uniform(0, 2 * np.pi)
    offset_height = np.random.uniform(-1, 1)
    
    x = cluster_x + np.cos(offset_angle) * np.sqrt(1 - offset_height**2) * offset_radius
    y = cluster_y + offset_height * offset_radius
    z = cluster_z + np.sin(offset_angle) * np.sqrt(1 - offset_height**2) * offset_radius
    
    return (float(x), float(y), float(z))


def recenter_positions(positions: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """
    Recenter a list of positions to have mean at origin
    
    Args:
        positions: List of (x, y, z) tuples
        
    Returns:
        Recentered positions
    """
    if not positions:
        return positions
    
    positions_array = np.array(positions)
    mean = positions_array.mean(axis=0)
    centered = positions_array - mean
    
    return [(float(x), float(y), float(z)) for x, y, z in centered]
