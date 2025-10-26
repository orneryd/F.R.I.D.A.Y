# Visualization Data Export Implementation

## Overview

The visualization data export feature provides REST API endpoints for exporting neuron network data in formats suitable for 3D rendering engines like Three.js. This enables real-time visualization of the neuron network structure, activation patterns, and cluster boundaries.

## Implementation Summary

### Files Created/Modified

1. **neuron_system/api/routes/visualization.py** (NEW)
   - Complete visualization endpoints implementation
   - Color and size calculation utilities
   - Three.js format conversion

2. **neuron_system/api/models.py** (MODIFIED)
   - Added visualization response models
   - Added Three.js compatible models

3. **neuron_system/api/app.py** (MODIFIED)
   - Registered visualization router

4. **test_visualization.py** (NEW)
   - Comprehensive test suite for all endpoints

5. **example_visualization_usage.py** (NEW)
   - Usage examples and Three.js integration guide

## API Endpoints

### 1. GET /api/v1/visualization/neurons

Export neuron positions and metadata for visualization.

**Query Parameters:**
- `neuron_type` (optional): Filter by neuron type (e.g., "knowledge", "tool")
- `min_activation` (optional): Minimum activation level (0.0 to 1.0)
- `max_activation` (optional): Maximum activation level (0.0 to 1.0)
- `limit` (optional): Maximum number of neurons to return

**Response:**
```json
{
  "neurons": [
    {
      "id": "uuid",
      "neuron_type": "knowledge",
      "position": {"x": 10.5, "y": 20.3, "z": -5.2},
      "activation_level": 0.75,
      "metadata": {},
      "color": "#6496ff",
      "size": 1.5,
      "label": "knowledge_abc12345"
    }
  ],
  "count": 100,
  "bounds": {
    "min": {"x": -100, "y": -100, "z": -100},
    "max": {"x": 100, "y": 100, "z": 100}
  }
}
```

**Features:**
- Automatic color assignment based on neuron type and activation
- Size calculation based on connection count and activation
- Bounding box calculation for camera positioning

### 2. GET /api/v1/visualization/synapses

Export synapse connections for visualization.

**Query Parameters:**
- `source_neuron_id` (optional): Filter by source neuron
- `target_neuron_id` (optional): Filter by target neuron
- `min_weight` (optional): Minimum synapse weight (-1.0 to 1.0)
- `max_weight` (optional): Maximum synapse weight (-1.0 to 1.0)
- `synapse_type` (optional): Filter by synapse type
- `limit` (optional): Maximum number of synapses to return

**Response:**
```json
{
  "synapses": [
    {
      "id": "uuid",
      "source_neuron_id": "uuid1",
      "target_neuron_id": "uuid2",
      "weight": 0.8,
      "synapse_type": "KNOWLEDGE",
      "color": "#00cc00",
      "thickness": 1.6
    }
  ],
  "count": 500
}
```

**Features:**
- Color gradient based on weight (green for positive, red for negative)
- Thickness proportional to weight magnitude
- Efficient filtering for large networks

### 3. GET /api/v1/visualization/clusters

Export cluster boundaries for visualization.

**Query Parameters:**
- `cluster_name` (optional): Filter by cluster name (partial match)

**Response:**
```json
{
  "clusters": [
    {
      "cluster_id": "uuid",
      "name": "data_processing_cluster",
      "neurons": ["uuid1", "uuid2", "uuid3"],
      "boundary": {
        "center": {"x": 15.0, "y": 20.0, "z": 10.0},
        "radius": 25.5,
        "neuron_count": 3
      },
      "metadata": {}
    }
  ],
  "count": 5
}
```

**Features:**
- Automatic bounding sphere calculation
- Centroid-based positioning
- Radius based on maximum distance from center

### 4. GET /api/v1/visualization/threejs

Export complete scene in Three.js compatible format.

**Query Parameters:**
- `neuron_type` (optional): Filter neurons by type
- `min_activation` (optional): Minimum activation level
- `include_synapses` (optional, default: true): Include synapse connections
- `include_clusters` (optional, default: true): Include cluster boundaries
- `max_neurons` (optional, default: 10000): Maximum neurons to export
- `max_synapses` (optional, default: 50000): Maximum synapses to export

**Response:**
```json
{
  "neurons": [
    {
      "id": "uuid",
      "position": {"x": 10.5, "y": 20.3, "z": -5.2},
      "color": {"r": 0.392, "g": 0.588, "b": 1.0},
      "size": 1.5,
      "type": "knowledge",
      "activation": 0.75,
      "metadata": {}
    }
  ],
  "synapses": [
    {
      "id": "uuid",
      "start": {"x": 10.5, "y": 20.3, "z": -5.2},
      "end": {"x": 15.0, "y": 25.0, "z": -3.0},
      "color": {"r": 0.0, "g": 0.8, "b": 0.0},
      "thickness": 1.6,
      "weight": 0.8,
      "type": "KNOWLEDGE"
    }
  ],
  "clusters": [
    {
      "id": "uuid",
      "name": "cluster_name",
      "center": {"x": 15.0, "y": 20.0, "z": 10.0},
      "radius": 25.5,
      "color": {"r": 0.2, "g": 0.6, "b": 1.0},
      "neuronCount": 3,
      "metadata": {}
    }
  ],
  "bounds": {
    "min": {"x": -100, "y": -100, "z": -100},
    "max": {"x": 100, "y": 100, "z": 100},
    "center": {"x": 0, "y": 0, "z": 0},
    "size": {"x": 200, "y": 200, "z": 200}
  },
  "metadata": {
    "neuronCount": 100,
    "synapseCount": 500,
    "clusterCount": 5,
    "format": "threejs",
    "version": "1.0.0"
  }
}
```

**Features:**
- Complete scene data in single request
- Normalized RGB colors (0.0 to 1.0) for Three.js
- Optimized for direct Three.js rendering
- Includes scene bounds for camera setup

## Visual Properties

### Neuron Colors

Colors are assigned based on neuron type with brightness modulated by activation level:

- **Knowledge Neurons**: Blue (#6496ff base)
- **Tool Neurons**: Orange (#ff9664 base)
- **Memory Neurons**: Green (#96ff64 base)
- **Sensor Neurons**: Yellow (#ffff64 base)
- **Decision Neurons**: Magenta (#ff64ff base)
- **Unknown Types**: Gray (#c8c8c8 base)

Activation level affects brightness:
- Low activation (0.0): 50% brightness
- High activation (1.0): 100% brightness

### Neuron Sizes

Size is calculated based on:
1. **Connection count**: More connections = larger size (up to 3x)
2. **Activation level**: Higher activation = slightly larger (0.8x to 1.2x)

Formula: `size = base_size * connection_factor * activation_factor`

### Synapse Colors

- **Positive weights**: Green gradient (darker = weaker, brighter = stronger)
- **Negative weights**: Red gradient (darker = weaker, brighter = stronger)

### Synapse Thickness

Proportional to weight magnitude:
- Minimum: 0.1 (weight = 0.0)
- Maximum: 2.0 (weight = ±1.0)

### Cluster Colors

Distinct colors assigned sequentially from a predefined palette:
1. Blue (0.2, 0.6, 1.0)
2. Orange (1.0, 0.6, 0.2)
3. Green (0.6, 1.0, 0.2)
4. Pink (1.0, 0.2, 0.6)
5. Purple (0.6, 0.2, 1.0)
6. Yellow (1.0, 1.0, 0.2)
7. Cyan (0.2, 1.0, 1.0)
8. Red (1.0, 0.2, 0.2)

## Three.js Integration

### Basic Setup

```javascript
// Fetch scene data
fetch('/api/v1/visualization/threejs')
    .then(response => response.json())
    .then(scene => {
        // Setup Three.js scene
        const threeScene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        
        // Position camera based on bounds
        const bounds = scene.bounds;
        camera.position.set(
            bounds.center.x,
            bounds.center.y,
            bounds.center.z + bounds.size.z * 2
        );
        
        // Render neurons, synapses, and clusters
        renderNeurons(threeScene, scene.neurons);
        renderSynapses(threeScene, scene.synapses);
        renderClusters(threeScene, scene.clusters);
    });
```

### Rendering Neurons

```javascript
function renderNeurons(scene, neurons) {
    neurons.forEach(neuron => {
        const geometry = new THREE.SphereGeometry(neuron.size, 16, 16);
        const material = new THREE.MeshPhongMaterial({
            color: new THREE.Color(neuron.color.r, neuron.color.g, neuron.color.b),
            emissive: new THREE.Color(neuron.color.r * 0.2, neuron.color.g * 0.2, neuron.color.b * 0.2)
        });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(neuron.position.x, neuron.position.y, neuron.position.z);
        sphere.userData = { id: neuron.id, type: neuron.type };
        scene.add(sphere);
    });
}
```

### Rendering Synapses

```javascript
function renderSynapses(scene, synapses) {
    synapses.forEach(synapse => {
        const points = [
            new THREE.Vector3(synapse.start.x, synapse.start.y, synapse.start.z),
            new THREE.Vector3(synapse.end.x, synapse.end.y, synapse.end.z)
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: new THREE.Color(synapse.color.r, synapse.color.g, synapse.color.b),
            linewidth: synapse.thickness,
            transparent: true,
            opacity: 0.6
        });
        const line = new THREE.Line(geometry, material);
        scene.add(line);
    });
}
```

### Rendering Clusters

```javascript
function renderClusters(scene, clusters) {
    clusters.forEach(cluster => {
        const geometry = new THREE.SphereGeometry(cluster.radius, 32, 32);
        const material = new THREE.MeshBasicMaterial({
            color: new THREE.Color(cluster.color.r, cluster.color.g, cluster.color.b),
            wireframe: true,
            transparent: true,
            opacity: 0.2
        });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(cluster.center.x, cluster.center.y, cluster.center.z);
        scene.add(sphere);
    });
}
```

## Performance Considerations

### Optimization Strategies

1. **Limit Parameters**: Use `max_neurons` and `max_synapses` to control data size
2. **Filtering**: Apply filters to reduce data transfer
3. **Progressive Loading**: Load neurons first, then synapses and clusters
4. **Level of Detail**: Adjust detail based on camera distance
5. **Culling**: Only render visible objects in Three.js

### Recommended Limits

- **Small networks** (< 1,000 neurons): No limits needed
- **Medium networks** (1,000 - 10,000 neurons): Limit synapses to 10,000
- **Large networks** (> 10,000 neurons): Use filtering and progressive loading

## Testing

Run the test suite:

```bash
python -m pytest test_visualization.py -v
```

Test coverage:
- ✓ Neuron export with filters
- ✓ Synapse export with filters
- ✓ Cluster export
- ✓ Three.js scene export
- ✓ Color calculations
- ✓ Size calculations
- ✓ Boundary calculations

## Usage Examples

See `example_visualization_usage.py` for complete examples including:
- Basic endpoint usage
- Filtering options
- Three.js integration
- Scene export and loading

Run the example:

```bash
# Start the API server
python run_api.py

# In another terminal
python example_visualization_usage.py
```

## Requirements Satisfied

✅ **Requirement 6.1**: Expose neuron positions and connections via query API  
✅ **Requirement 6.2**: Provide metadata for each neuron including activation level and connection count  
✅ **Requirement 6.3**: Return data in 3D-rendering-compatible format (Three.js)  
✅ **Requirement 6.4**: Support filtering visualization data by activation threshold  
✅ **Requirement 6.5**: Calculate and expose cluster information for neuron groups  

## Future Enhancements

1. **WebSocket Support**: Real-time updates for live visualization
2. **Animation Data**: Export activation propagation sequences
3. **Custom Color Schemes**: User-defined color mappings
4. **LOD Support**: Multiple detail levels for large networks
5. **Spatial Partitioning**: Export octree structure for efficient rendering
6. **VR/AR Support**: Additional formats for immersive visualization
