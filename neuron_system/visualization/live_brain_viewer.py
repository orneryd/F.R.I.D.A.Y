"""
Live 3D Brain Visualization Server
Shows real-time updates as neurons are created and trained
"""
import json
import threading
import time
from pathlib import Path
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

from neuron_system.storage.neuron_store import NeuronStore
from neuron_system.storage.synapse_store import SynapseStore


class LiveBrainViewer:
    """Real-time 3D visualization of the neural network"""
    
    def __init__(self, neuron_store: NeuronStore, synapse_store: SynapseStore, port: int = 5000):
        self.neuron_store = neuron_store
        self.synapse_store = synapse_store
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'friday-brain-viewer'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self.monitoring = False
        self.monitor_thread = None
        self.last_neuron_count = 0
        self.last_synapse_count = 0
        
        # Default filtering settings (can be overridden)
        self.default_max_synapses = 1000
        self.default_min_weight = 0.3
        
        self._setup_routes()
        self._setup_socketio()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('brain_viewer.html')
        
        @self.app.route('/api/brain_data')
        def get_brain_data():
            from flask import request
            max_synapses = int(request.args.get('max_synapses', self.default_max_synapses))
            min_weight = float(request.args.get('min_weight', self.default_min_weight))
            data = self._get_brain_data(max_synapses=max_synapses, min_weight=min_weight)
            return jsonify(data)
    
    def _setup_socketio(self):
        """Setup WebSocket handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print("Client connected")
            # Send initial data with default settings
            data = self._get_brain_data(
                max_synapses=self.default_max_synapses,
                min_weight=self.default_min_weight
            )
            emit('brain_update', data)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print("Client disconnected")
        
        @self.socketio.on('start_monitoring')
        def handle_start_monitoring():
            self.start_monitoring()
            emit('monitoring_started', {'status': 'monitoring'})
        
        @self.socketio.on('stop_monitoring')
        def handle_stop_monitoring():
            self.stop_monitoring()
            emit('monitoring_stopped', {'status': 'stopped'})
    
    def _get_brain_data(self, max_synapses=1000, min_weight=0.3):
        """
        Get current brain state as JSON with performance optimizations.
        
        Args:
            max_synapses: Maximum number of synapses to show
            min_weight: Minimum weight threshold for synapses
        """
        neurons = self.neuron_store.list_all()
        
        # OPTIMIZED: Query synapses directly with SQL filter
        import sqlite3
        synapses = []
        
        with self.synapse_store.db.get_connection() as conn:
            # Get only strong synapses, sorted by weight, limited
            cursor = conn.execute("""
                SELECT * FROM synapses 
                WHERE ABS(weight) >= ?
                ORDER BY ABS(weight) DESC
                LIMIT ?
            """, (min_weight, max_synapses))
            
            rows = cursor.fetchall()
            
            # Convert to Synapse objects
            from neuron_system.core.synapse import Synapse
            for row in rows:
                row_dict = dict(row)
                synapse = self.synapse_store._row_to_synapse(row_dict)
                synapses.append(synapse)
        
        print(f"Loaded {len(synapses)} synapses (weight >= {min_weight}) in optimized query")
        
        # Create neuron position lookup and collect all positions
        neuron_positions = {}
        all_positions = []
        for neuron in neurons:
            pos = neuron.to_dict().get('position', {'x': 0, 'y': 0, 'z': 0})
            position = (pos.get('x', 0), pos.get('y', 0), pos.get('z', 0))
            neuron_positions[str(neuron.id)] = position
            all_positions.append(position)
        
        # Calculate center and recenter all positions
        if all_positions:
            import numpy as np
            positions_array = np.array(all_positions)
            center = positions_array.mean(axis=0)
            
            # Recenter positions
            centered_positions = {}
            for neuron_id, pos in neuron_positions.items():
                centered_pos = (
                    pos[0] - center[0],
                    pos[1] - center[1],
                    pos[2] - center[2]
                )
                centered_positions[neuron_id] = centered_pos
            neuron_positions = centered_positions
        
        # Count connections per neuron
        connection_counts = {}
        for synapse in synapses:
            source_id = str(synapse.source_neuron_id)
            target_id = str(synapse.target_neuron_id)
            connection_counts[source_id] = connection_counts.get(source_id, 0) + 1
            connection_counts[target_id] = connection_counts.get(target_id, 0) + 1
        
        # Prepare neuron data
        neuron_data = []
        neurons_without_position = 0
        
        for neuron in neurons:
            neuron_dict = neuron.to_dict()
            pos = neuron_dict.get('position', {'x': 0, 'y': 0, 'z': 0})
            
            # Check if position is valid (not all zeros)
            if pos.get('x', 0) == 0 and pos.get('y', 0) == 0 and pos.get('z', 0) == 0:
                neurons_without_position += 1
            
            metadata = neuron_dict.get('metadata', {})
            neuron_id = str(neuron.id)
            
            # Get tags from semantic_tags or metadata
            tags = neuron_dict.get('semantic_tags', metadata.get('tags', []))
            if not isinstance(tags, list):
                tags = []
            
            # Get content from source_data or metadata
            content = neuron_dict.get('source_data', metadata.get('content', ''))
            if content and len(str(content)) > 100:
                content = str(content)[:100] + '...'
            
            # Use CENTERED position from neuron_positions
            centered_pos = neuron_positions.get(neuron_id, (0, 0, 0))
            
            neuron_data.append({
                'id': neuron_id,
                'x': centered_pos[0],
                'y': centered_pos[1],
                'z': centered_pos[2],
                'tags': tags,
                'content': str(content),
                'activation': neuron.activation_level,
                'connections': connection_counts.get(neuron_id, 0)
            })
        
        if neurons_without_position > 0:
            print(f"⚠️  Warning: {neurons_without_position}/{len(neurons)} neurons have no position (0,0,0)")
        
        # Prepare synapse data
        synapse_data = []
        for synapse in synapses:
            source_id = str(synapse.source_neuron_id)
            target_id = str(synapse.target_neuron_id)
            
            if source_id in neuron_positions and target_id in neuron_positions:
                source_pos = neuron_positions[source_id]
                target_pos = neuron_positions[target_id]
                
                synapse_data.append({
                    'source': list(source_pos),
                    'target': list(target_pos),
                    'weight': abs(synapse.weight)  # Use absolute value for visibility
                })
        
        return {
            'neurons': neuron_data,
            'synapses': synapse_data,
            'stats': {
                'total_neurons': len(neurons),
                'total_synapses': len(synapse_data),
                'avg_connections': len(synapse_data) / len(neurons) if neurons else 0
            }
        }
    
    def start_monitoring(self):
        """Start monitoring for changes"""
        if not self.monitoring:
            self.monitoring = True
            self.last_neuron_count = len(self.neuron_store.list_all())
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("Started monitoring brain changes")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print("Stopped monitoring")
    
    def _monitor_loop(self):
        """Monitor for changes and broadcast updates"""
        while self.monitoring:
            try:
                # Force fresh connection to see latest data
                with self.neuron_store.db.get_connection() as conn:
                    cursor = conn.execute("SELECT COUNT(*) as count FROM neurons")
                    current_neuron_count = cursor.fetchone()['count']
                    
                    cursor = conn.execute("SELECT COUNT(*) as count FROM synapses")
                    current_synapse_count = cursor.fetchone()['count']
                
                # Check if anything changed
                if (current_neuron_count != self.last_neuron_count or 
                    current_synapse_count != self.last_synapse_count):
                    
                    neuron_diff = current_neuron_count - self.last_neuron_count
                    synapse_diff = current_synapse_count - self.last_synapse_count
                    
                    if neuron_diff > 0:
                        print(f"🧠 +{neuron_diff} neuron(s) created")
                    if synapse_diff > 0:
                        print(f"🔗 +{synapse_diff} synapse(s) formed")
                    elif synapse_diff < 0:
                        print(f"✂️  {abs(synapse_diff)} weak synapse(s) pruned")
                    
                    # Get updated data
                    data = self._get_brain_data()
                    
                    # Broadcast to all connected clients
                    self.socketio.emit('brain_update', data)
                    
                    self.last_neuron_count = current_neuron_count
                    self.last_synapse_count = current_synapse_count
                
                time.sleep(0.2)  # Check every 200ms for faster updates
                
            except Exception as e:
                print(f"Error in monitor loop: {e}")
                time.sleep(1)
    
    def run(self, debug=False):
        """Start the live viewer server"""
        print(f"Starting Live Brain Viewer on http://localhost:{self.port}")
        print("Open your browser to see real-time visualization")
        self.start_monitoring()
        self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=debug)
