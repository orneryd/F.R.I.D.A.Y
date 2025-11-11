"""
Brain Viewer - 3D Visualization of Friday's Neural Network.

Interactive 3D visualization showing:
- Neurons as points in 3D space
- Synapses as connections
- Color-coded by type/tags
- Interactive exploration
"""

import logging
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class BrainViewer:
    """
    3D visualization of Friday's neural network.
    
    Shows neurons in 3D space with their connections.
    """
    
    def __init__(self, graph):
        """
        Initialize brain viewer.
        
        Args:
            graph: NeuronGraph instance
        """
        self.graph = graph
        self.fig = None
    
    def visualize(
        self,
        max_neurons: Optional[int] = 1000,
        show_synapses: bool = True,
        color_by: str = "type",
        title: str = "Friday's Brain - 3D Neural Network"
    ):
        """
        Create 3D visualization of neural network.
        
        Args:
            max_neurons: Maximum neurons to show (for performance)
            show_synapses: Whether to show connections
            color_by: Color neurons by 'type', 'activation', or 'tags'
            title: Plot title
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("Plotly not installed. Run: pip install plotly")
            return None
        
        logger.info(f"Visualizing {len(self.graph.neurons)} neurons...")
        
        # Get neurons (limit for performance)
        neurons = list(self.graph.neurons.values())[:max_neurons]
        
        if not neurons:
            logger.warning("No neurons to visualize")
            return None
        
        # Extract positions and metadata
        positions = []
        colors = []
        texts = []
        sizes = []
        
        for neuron in neurons:
            if hasattr(neuron, 'position'):
                pos = neuron.position
                positions.append([pos.x, pos.y, pos.z])
                
                # Color
                color = self._get_neuron_color(neuron, color_by)
                colors.append(color)
                
                # Hover text
                text = self._get_neuron_info(neuron)
                texts.append(text)
                
                # Size (based on connections)
                size = self._get_neuron_size(neuron)
                sizes.append(size)
        
        if not positions:
            logger.warning("No neurons with positions found")
            return None
        
        positions = np.array(positions)
        
        # Create 3D scatter plot
        neuron_trace = go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_by.capitalize()),
                line=dict(color='white', width=0.5)
            ),
            text=texts,
            hoverinfo='text',
            name='Neurons'
        )
        
        data = [neuron_trace]
        
        # Add synapses if requested
        if show_synapses:
            synapse_traces = self._create_synapse_traces(neurons, positions)
            data.extend(synapse_traces)
        
        # Create layout
        layout = go.Layout(
            title=dict(
                text=title,
                font=dict(size=20)
            ),
            scene=dict(
                xaxis=dict(title='X', backgroundcolor="rgb(230, 230,230)"),
                yaxis=dict(title='Y', backgroundcolor="rgb(230, 230,230)"),
                zaxis=dict(title='Z', backgroundcolor="rgb(230, 230,230)"),
                bgcolor="rgb(240, 240, 240)"
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        self.fig = go.Figure(data=data, layout=layout)
        
        logger.info("✓ Visualization created")
        return self.fig
    
    def show(self):
        """Show the visualization in browser."""
        if self.fig:
            self.fig.show()
        else:
            logger.warning("No visualization created. Call visualize() first.")
    
    def save(self, filename: str = "friday_brain.html"):
        """Save visualization to HTML file."""
        if self.fig:
            self.fig.write_html(filename)
            logger.info(f"✓ Saved to {filename}")
        else:
            logger.warning("No visualization created. Call visualize() first.")
    
    def _get_neuron_color(self, neuron, color_by: str) -> float:
        """Get color value for neuron."""
        if color_by == "type":
            # Color by neuron type
            type_map = {
                'knowledge': 0,
                'memory': 1,
                'reasoning': 2,
                'logic': 3,
                'pattern': 4
            }
            
            if hasattr(neuron, 'semantic_tags'):
                for tag in neuron.semantic_tags:
                    if tag in type_map:
                        return type_map[tag]
            return 5  # Unknown
        
        elif color_by == "activation":
            # Color by activation level
            return getattr(neuron, 'activation', 0.5)
        
        else:
            # Default
            return 0.5
    
    def _get_neuron_info(self, neuron) -> str:
        """Get hover info for neuron."""
        info = []
        
        # ID
        if hasattr(neuron, 'id'):
            info.append(f"ID: {str(neuron.id)[:8]}...")
        
        # Type/Tags
        if hasattr(neuron, 'semantic_tags'):
            tags = ', '.join(neuron.semantic_tags[:3])
            info.append(f"Tags: {tags}")
        
        # Content preview
        if hasattr(neuron, 'source_data'):
            preview = neuron.source_data[:50]
            info.append(f"Content: {preview}...")
        
        # Connections
        if hasattr(neuron, 'synapses'):
            info.append(f"Connections: {len(neuron.synapses)}")
        
        return '<br>'.join(info)
    
    def _get_neuron_size(self, neuron) -> float:
        """Get display size for neuron based on connections."""
        base_size = 5
        
        if hasattr(neuron, 'synapses'):
            # More connections = bigger
            connection_bonus = min(len(neuron.synapses) * 0.5, 10)
            return base_size + connection_bonus
        
        return base_size
    
    def _create_synapse_traces(
        self,
        neurons: List,
        positions: np.ndarray
    ) -> List:
        """Create traces for synapses (connections)."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return []
        
        # Build neuron ID to index map
        neuron_map = {str(n.id): i for i, n in enumerate(neurons)}
        
        # Collect synapse lines
        x_lines = []
        y_lines = []
        z_lines = []
        
        synapse_count = 0
        max_synapses = 500  # Limit for performance
        
        for i, neuron in enumerate(neurons):
            if not hasattr(neuron, 'synapses'):
                continue
            
            for synapse in neuron.synapses:
                if synapse_count >= max_synapses:
                    break
                
                target_id = str(synapse.target_id)
                if target_id in neuron_map:
                    j = neuron_map[target_id]
                    
                    # Add line from neuron i to neuron j
                    x_lines.extend([positions[i, 0], positions[j, 0], None])
                    y_lines.extend([positions[i, 1], positions[j, 1], None])
                    z_lines.extend([positions[i, 2], positions[j, 2], None])
                    
                    synapse_count += 1
            
            if synapse_count >= max_synapses:
                break
        
        if not x_lines:
            return []
        
        # Create synapse trace
        synapse_trace = go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            line=dict(
                color='rgba(100, 100, 100, 0.2)',
                width=1
            ),
            hoverinfo='none',
            name=f'Synapses ({synapse_count})'
        )
        
        logger.info(f"  Added {synapse_count} synapses")
        
        return [synapse_trace]
    
    def visualize_clusters(
        self,
        max_neurons: int = 1000,
        title: str = "Friday's Brain - Clustered View"
    ):
        """
        Visualize neurons grouped by clusters/tags.
        
        Args:
            max_neurons: Maximum neurons to show
            title: Plot title
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.error("Plotly not installed. Run: pip install plotly")
            return None
        
        logger.info("Creating clustered visualization...")
        
        # Group neurons by primary tag
        clusters = {}
        neurons = list(self.graph.neurons.values())[:max_neurons]
        
        for neuron in neurons:
            if not hasattr(neuron, 'position'):
                continue
            
            # Get primary tag
            tag = 'unknown'
            if hasattr(neuron, 'semantic_tags') and neuron.semantic_tags:
                tag = neuron.semantic_tags[0]
            
            if tag not in clusters:
                clusters[tag] = []
            
            clusters[tag].append(neuron)
        
        # Create trace for each cluster
        data = []
        
        for tag, cluster_neurons in clusters.items():
            positions = []
            texts = []
            
            for neuron in cluster_neurons:
                pos = neuron.position
                positions.append([pos.x, pos.y, pos.z])
                texts.append(self._get_neuron_info(neuron))
            
            if not positions:
                continue
            
            positions = np.array(positions)
            
            trace = go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    line=dict(color='white', width=0.5)
                ),
                text=texts,
                hoverinfo='text',
                name=f'{tag} ({len(cluster_neurons)})'
            )
            
            data.append(trace)
        
        # Create layout
        layout = go.Layout(
            title=dict(text=title, font=dict(size=20)),
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z')
            ),
            showlegend=True,
            hovermode='closest'
        )
        
        self.fig = go.Figure(data=data, layout=layout)
        
        logger.info(f"✓ Created {len(clusters)} clusters")
        return self.fig
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the neural network."""
        stats = {
            "total_neurons": len(self.graph.neurons),
            "total_synapses": 0,
            "clusters": {},
            "avg_connections": 0
        }
        
        # Count synapses and clusters
        total_connections = 0
        
        for neuron in self.graph.neurons.values():
            if hasattr(neuron, 'synapses'):
                total_connections += len(neuron.synapses)
                stats["total_synapses"] += len(neuron.synapses)
            
            # Count by tag
            if hasattr(neuron, 'semantic_tags') and neuron.semantic_tags:
                tag = neuron.semantic_tags[0]
                stats["clusters"][tag] = stats["clusters"].get(tag, 0) + 1
        
        if stats["total_neurons"] > 0:
            stats["avg_connections"] = total_connections / stats["total_neurons"]
        
        return stats
