"""
Example usage of visualization endpoints for the 3D Synaptic Neuron System.

This demonstrates how to export neuron network data for 3D visualization
using Three.js or other rendering engines.
"""

import requests
import json


def main():
    """Demonstrate visualization endpoint usage"""
    
    # Base URL for the API
    base_url = "http://localhost:8000/api/v1"
    
    print("=" * 80)
    print("3D Synaptic Neuron System - Visualization Examples")
    print("=" * 80)
    
    # Example 1: Get all neurons for visualization
    print("\n1. Getting neuron visualization data...")
    response = requests.get(f"{base_url}/visualization/neurons")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Retrieved {data['count']} neurons")
        print(f"   ✓ Bounds: {data['bounds']}")
        
        if data['neurons']:
            neuron = data['neurons'][0]
            print(f"   ✓ Sample neuron:")
            print(f"     - ID: {neuron['id']}")
            print(f"     - Type: {neuron['neuron_type']}")
            print(f"     - Position: ({neuron['position']['x']:.2f}, "
                  f"{neuron['position']['y']:.2f}, {neuron['position']['z']:.2f})")
            print(f"     - Color: {neuron['color']}")
            print(f"     - Size: {neuron['size']:.2f}")
            print(f"     - Activation: {neuron['activation_level']:.2f}")
    else:
        print(f"   ✗ Error: {response.status_code}")
    
    # Example 2: Get neurons with filters
    print("\n2. Getting filtered neurons (knowledge type, high activation)...")
    response = requests.get(
        f"{base_url}/visualization/neurons",
        params={
            "neuron_type": "knowledge",
            "min_activation": 0.5,
            "limit": 100
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Retrieved {data['count']} filtered neurons")
    else:
        print(f"   ✗ Error: {response.status_code}")
    
    # Example 3: Get synapse connections
    print("\n3. Getting synapse visualization data...")
    response = requests.get(f"{base_url}/visualization/synapses")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Retrieved {data['count']} synapses")
        
        if data['synapses']:
            synapse = data['synapses'][0]
            print(f"   ✓ Sample synapse:")
            print(f"     - ID: {synapse['id']}")
            print(f"     - Source → Target: {synapse['source_neuron_id'][:8]}... → "
                  f"{synapse['target_neuron_id'][:8]}...")
            print(f"     - Weight: {synapse['weight']:.2f}")
            print(f"     - Color: {synapse['color']}")
            print(f"     - Thickness: {synapse['thickness']:.2f}")
    else:
        print(f"   ✗ Error: {response.status_code}")
    
    # Example 4: Get synapses with filters
    print("\n4. Getting filtered synapses (strong connections only)...")
    response = requests.get(
        f"{base_url}/visualization/synapses",
        params={
            "min_weight": 0.7,
            "limit": 50
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Retrieved {data['count']} strong synapses")
    else:
        print(f"   ✗ Error: {response.status_code}")
    
    # Example 5: Get cluster boundaries
    print("\n5. Getting cluster visualization data...")
    response = requests.get(f"{base_url}/visualization/clusters")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Retrieved {data['count']} clusters")
        
        if data['clusters']:
            cluster = data['clusters'][0]
            print(f"   ✓ Sample cluster:")
            print(f"     - Name: {cluster['name']}")
            print(f"     - Neurons: {len(cluster['neurons'])}")
            print(f"     - Center: ({cluster['boundary']['center']['x']:.2f}, "
                  f"{cluster['boundary']['center']['y']:.2f}, "
                  f"{cluster['boundary']['center']['z']:.2f})")
            print(f"     - Radius: {cluster['boundary']['radius']:.2f}")
    else:
        print(f"   ✗ Error: {response.status_code}")
    
    # Example 6: Get complete Three.js scene
    print("\n6. Getting complete Three.js scene...")
    response = requests.get(
        f"{base_url}/visualization/threejs",
        params={
            "max_neurons": 1000,
            "max_synapses": 5000,
            "include_synapses": True,
            "include_clusters": True
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Retrieved complete scene:")
        print(f"     - Neurons: {data['metadata']['neuronCount']}")
        print(f"     - Synapses: {data['metadata']['synapseCount']}")
        print(f"     - Clusters: {data['metadata']['clusterCount']}")
        print(f"     - Format: {data['metadata']['format']}")
        print(f"     - Bounds: {data['bounds']}")
        
        # Show sample Three.js neuron format
        if data['neurons']:
            neuron = data['neurons'][0]
            print(f"\n   ✓ Three.js neuron format:")
            print(f"     - Position: {{x: {neuron['position']['x']:.2f}, "
                  f"y: {neuron['position']['y']:.2f}, z: {neuron['position']['z']:.2f}}}")
            print(f"     - Color: {{r: {neuron['color']['r']:.3f}, "
                  f"g: {neuron['color']['g']:.3f}, b: {neuron['color']['b']:.3f}}}")
            print(f"     - Size: {neuron['size']:.2f}")
        
        # Show sample Three.js synapse format
        if data['synapses']:
            synapse = data['synapses'][0]
            print(f"\n   ✓ Three.js synapse format:")
            print(f"     - Start: {{x: {synapse['start']['x']:.2f}, "
                  f"y: {synapse['start']['y']:.2f}, z: {synapse['start']['z']:.2f}}}")
            print(f"     - End: {{x: {synapse['end']['x']:.2f}, "
                  f"y: {synapse['end']['y']:.2f}, z: {synapse['end']['z']:.2f}}}")
            print(f"     - Color: {{r: {synapse['color']['r']:.3f}, "
                  f"g: {synapse['color']['g']:.3f}, b: {synapse['color']['b']:.3f}}}")
        
        # Save to file for Three.js usage
        with open("threejs_scene.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n   ✓ Scene saved to 'threejs_scene.json'")
        print(f"     You can now load this file in your Three.js application!")
    else:
        print(f"   ✗ Error: {response.status_code}")
    
    # Example 7: Integration example for Three.js
    print("\n7. Three.js Integration Example:")
    print("""
    // JavaScript code to load and render the scene in Three.js:
    
    fetch('/api/v1/visualization/threejs')
        .then(response => response.json())
        .then(scene => {
            // Create neurons as spheres
            scene.neurons.forEach(neuron => {
                const geometry = new THREE.SphereGeometry(neuron.size, 16, 16);
                const material = new THREE.MeshPhongMaterial({
                    color: new THREE.Color(
                        neuron.color.r,
                        neuron.color.g,
                        neuron.color.b
                    )
                });
                const sphere = new THREE.Mesh(geometry, material);
                sphere.position.set(
                    neuron.position.x,
                    neuron.position.y,
                    neuron.position.z
                );
                threeScene.add(sphere);
            });
            
            // Create synapses as lines
            scene.synapses.forEach(synapse => {
                const points = [
                    new THREE.Vector3(
                        synapse.start.x,
                        synapse.start.y,
                        synapse.start.z
                    ),
                    new THREE.Vector3(
                        synapse.end.x,
                        synapse.end.y,
                        synapse.end.z
                    )
                ];
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const material = new THREE.LineBasicMaterial({
                    color: new THREE.Color(
                        synapse.color.r,
                        synapse.color.g,
                        synapse.color.b
                    ),
                    linewidth: synapse.thickness
                });
                const line = new THREE.Line(geometry, material);
                threeScene.add(line);
            });
            
            // Create cluster boundaries as wireframe spheres
            scene.clusters.forEach(cluster => {
                const geometry = new THREE.SphereGeometry(cluster.radius, 16, 16);
                const material = new THREE.MeshBasicMaterial({
                    color: new THREE.Color(
                        cluster.color.r,
                        cluster.color.g,
                        cluster.color.b
                    ),
                    wireframe: true,
                    transparent: true,
                    opacity: 0.3
                });
                const sphere = new THREE.Mesh(geometry, material);
                sphere.position.set(
                    cluster.center.x,
                    cluster.center.y,
                    cluster.center.z
                );
                threeScene.add(sphere);
            });
        });
    """)
    
    print("\n" + "=" * 80)
    print("Visualization examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    print("\nNote: Make sure the API server is running on http://localhost:8000")
    print("Start it with: python run_api.py\n")
    
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API server")
        print("  Please start the server with: python run_api.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
