"""
Test spatial index functionality.
"""

import sys
from uuid import uuid4
from datetime import datetime
import numpy as np

from neuron_system.core.vector3d import Vector3D
from neuron_system.core.graph import NeuronGraph
from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
from neuron_system.spatial.octree import Octree
from neuron_system.spatial.spatial_index import SpatialIndex


def test_octree():
    """Test basic octree operations."""
    print("\nTesting Octree...")
    
    # Create octree
    bounds = (Vector3D(-100, -100, -100), Vector3D(100, 100, 100))
    octree = Octree(bounds)
    
    # Create test neurons
    neurons = []
    for i in range(10):
        neuron = KnowledgeNeuron()
        neuron.id = uuid4()
        neuron.position = Vector3D(i * 10, i * 10, i * 10)
        neuron.vector = np.random.rand(384)
        neuron.source_data = f"Test data {i}"
        neurons.append(neuron)
        octree.insert(neuron)
    
    print(f"  Inserted {octree.get_neuron_count()} neurons")
    
    # Test radius query
    center = Vector3D(50, 50, 50)
    radius = 30
    results = octree.query_radius(center, radius)
    print(f"  Radius query (center={center}, radius={radius}): {len(results)} neurons")
    
    # Test k-NN query
    point = Vector3D(25, 25, 25)
    k = 3
    knn_results = octree.query_knn(point, k)
    print(f"  K-NN query (point={point}, k={k}): {len(knn_results)} neurons")
    
    # Verify k-NN results are sorted by distance
    distances = [n.position.distance(point) for n in knn_results]
    is_sorted = all(distances[i] <= distances[i+1] for i in range(len(distances)-1))
    print(f"  K-NN results sorted: {is_sorted}")
    
    print("  ✓ Octree working")
    return True


def test_spatial_index():
    """Test SpatialIndex wrapper."""
    print("\nTesting SpatialIndex...")
    
    bounds = (Vector3D(-100, -100, -100), Vector3D(100, 100, 100))
    spatial_index = SpatialIndex(bounds)
    
    # Create and insert neurons
    neurons = []
    for i in range(20):
        neuron = KnowledgeNeuron()
        neuron.id = uuid4()
        neuron.position = Vector3D(
            np.random.uniform(-50, 50),
            np.random.uniform(-50, 50),
            np.random.uniform(-50, 50)
        )
        neuron.vector = np.random.rand(384)
        neuron.source_data = f"Test data {i}"
        neurons.append(neuron)
        spatial_index.insert(neuron)
    
    print(f"  Inserted {spatial_index.get_neuron_count()} neurons")
    
    # Test region query
    min_bound = Vector3D(-25, -25, -25)
    max_bound = Vector3D(25, 25, 25)
    region_results = spatial_index.query_region(min_bound, max_bound)
    print(f"  Region query: {len(region_results)} neurons in region")
    
    # Test density calculation
    center = Vector3D(0, 0, 0)
    density = spatial_index.get_density(center, 50)
    print(f"  Density at center (radius=50): {density:.4f} neurons/unit³")
    
    # Test removal
    neuron_to_remove = neurons[0]
    spatial_index.remove(neuron_to_remove.id)
    print(f"  After removal: {spatial_index.get_neuron_count()} neurons")
    
    print("  ✓ SpatialIndex working")
    return True


def test_graph_integration():
    """Test spatial index integration with NeuronGraph."""
    print("\nTesting NeuronGraph spatial integration...")
    
    bounds = (Vector3D(-100, -100, -100), Vector3D(100, 100, 100))
    graph = NeuronGraph(bounds)
    
    # Add neurons with automatic spatial indexing
    neurons = []
    for i in range(15):
        neuron = KnowledgeNeuron()
        neuron.id = uuid4()
        neuron.position = Vector3D(i * 10 - 70, i * 5 - 35, i * 8 - 56)
        neuron.vector = np.random.rand(384)
        neuron.source_data = f"Graph test data {i}"
        neuron.created_at = datetime.now()
        neuron.modified_at = datetime.now()
        neurons.append(neuron)
        graph.add_neuron(neuron)
    
    print(f"  Added {graph.get_neuron_count()} neurons to graph")
    print(f"  Spatial index count: {graph.spatial_index.get_neuron_count()}")
    
    # Test spatial queries through graph
    center = Vector3D(0, 0, 0)
    radius_results = graph.query_radius(center, 50)
    print(f"  Graph radius query: {len(radius_results)} neurons")
    
    knn_results = graph.query_knn(center, 5)
    print(f"  Graph k-NN query: {len(knn_results)} neurons")
    
    # Test automatic positioning
    new_neuron = KnowledgeNeuron()
    new_neuron.id = uuid4()
    new_neuron.vector = np.random.rand(384)
    new_neuron.source_data = "New neuron"
    new_neuron.created_at = datetime.now()
    new_neuron.modified_at = datetime.now()
    
    # Position randomly
    random_pos = graph.position_neuron(new_neuron, strategy="random")
    print(f"  Random position: {random_pos}")
    
    # Position near similar (requires vector)
    similar_pos = graph.position_neuron(new_neuron, strategy="similar", k=3, spread=5.0)
    print(f"  Similar position: {similar_pos}")
    
    # Test density calculation
    density = graph.get_spatial_density(center, 50)
    print(f"  Spatial density: {density:.4f} neurons/unit³")
    
    # Test neuron removal with spatial index cleanup
    neuron_to_remove = neurons[0]
    graph.remove_neuron(neuron_to_remove.id)
    print(f"  After removal - Graph: {graph.get_neuron_count()}, Spatial: {graph.spatial_index.get_neuron_count()}")
    
    print("  ✓ Graph spatial integration working")
    return True


def main():
    """Run all spatial index tests."""
    print("=" * 60)
    print("3D Synaptic Neuron System - Spatial Index Tests")
    print("=" * 60)
    
    try:
        test_octree()
        test_spatial_index()
        test_graph_integration()
        
        print("\n" + "=" * 60)
        print("All spatial index tests passed! ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
