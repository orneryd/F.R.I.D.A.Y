"""
High-Performance 3D Brain Viewer using Three.js

This viewer can handle 100K+ synapses smoothly using WebGL.
Integrated into the main neuron_system package.
"""

import argparse
import threading
import time
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from neuron_system.storage.database import DatabaseManager
from neuron_system.storage.neuron_store import NeuronStore
from neuron_system.storage.synapse_store import SynapseStore


class ThreeJSBrainViewer:
    """High-performance 3D brain viewer using Three.js and WebGL."""
    
    def __init__(self, database_path='data/neuron_system.db'):
        self.database_path = database_path
        self.db_manager = DatabaseManager(database_path)
        self.neuron_store = NeuronStore(self.db_manager)
        self.synapse_store = SynapseStore(self.db_manager)
        
        # HuggingFace token storage
        self.hf_token = None
        self.hf_logged_in = False
        
        # Initialize training system
        self._init_training_system()
        
        # Flask app with SocketIO
        self.app = Flask(__name__, 
                        template_folder='templates')
        self.app.config['SECRET_KEY'] = 'friday-brain-viewer-secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Training state
        self.training_active = False
        self.training_thread = None
        
        self._setup_routes()
        self._setup_socketio()
    
    def _init_training_system(self):
        """Initialize the training system."""
        from neuron_system.core.graph import NeuronGraph
        from neuron_system.engines.compression import CompressionEngine
        from neuron_system.engines.query import QueryEngine
        from neuron_system.engines.training import TrainingEngine
        from neuron_system.ai.language_model import LanguageModel
        
        self.graph = NeuronGraph()
        self.graph.attach_storage(self.db_manager)
        
        try:
            self.graph.load()
        except:
            pass
        
        self.compression_engine = CompressionEngine()
        self.query_engine = QueryEngine(self.graph, self.compression_engine)
        self.training_engine = TrainingEngine(self.graph)
        self.language_model = LanguageModel(
            self.graph,
            self.compression_engine,
            self.query_engine,
            self.training_engine
        )
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('brain_viewer_threejs.html')
        
        @self.app.route('/api/neuron/<neuron_id>')
        def get_neuron_details(neuron_id):
            """Get detailed information about a specific neuron."""
            from uuid import UUID
            
            try:
                neuron = self.neuron_store.get(UUID(neuron_id))
                if neuron:
                    return jsonify(neuron.to_dict())
                else:
                    return jsonify({'error': 'Neuron not found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/brain_data')
        def get_brain_data():
            max_synapses = int(request.args.get('max_synapses', 1000))
            min_weight = float(request.args.get('min_weight', 0.3))
            
            print(f"Loading brain data: max_synapses={max_synapses}, min_weight={min_weight}")
            
            # Get neurons
            neurons = self.neuron_store.list_all()
            
            # Get synapses with SQL optimization
            synapses = []
            with self.synapse_store.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM synapses 
                    WHERE ABS(weight) >= ?
                    ORDER BY ABS(weight) DESC
                    LIMIT ?
                """, (min_weight, max_synapses))
                
                rows = cursor.fetchall()
                for row in rows:
                    row_dict = dict(row)
                    synapse = self.synapse_store._row_to_synapse(row_dict)
                    synapses.append(synapse)
            
            # Convert to JSON-friendly format and calculate center
            neurons_data = []
            positions = []
            
            for neuron in neurons:
                n_dict = neuron.to_dict()
                pos = {
                    'x': n_dict.get('position', {}).get('x', 0),
                    'y': n_dict.get('position', {}).get('y', 0),
                    'z': n_dict.get('position', {}).get('z', 0)
                }
                positions.append([pos['x'], pos['y'], pos['z']])
                neurons_data.append({
                    'id': str(neuron.id),
                    'position': pos,
                    'neuron_type': n_dict.get('neuron_type', 'KNOWLEDGE')
                })
            
            # Calculate center and recenter all neurons
            if positions:
                import numpy as np
                positions_array = np.array(positions)
                center = positions_array.mean(axis=0)
                
                # Recenter all neurons
                for i, neuron_data in enumerate(neurons_data):
                    neuron_data['position']['x'] -= center[0]
                    neuron_data['position']['y'] -= center[1]
                    neuron_data['position']['z'] -= center[2]
            
            synapses_data = []
            for synapse in synapses:
                synapses_data.append({
                    'source_neuron_id': str(synapse.source_neuron_id),
                    'target_neuron_id': str(synapse.target_neuron_id),
                    'weight': synapse.weight
                })
            
            print(f"Loaded {len(neurons_data)} neurons and {len(synapses_data)} synapses")
            
            return jsonify({
                'neurons': neurons_data,
                'synapses': synapses_data
            })
        
        @self.app.route('/api/hf/login', methods=['POST'])
        def hf_login():
            """Login to HuggingFace."""
            data = request.json
            token = data.get('token', '')
            
            if not token:
                return jsonify({'error': 'No token provided'}), 400
            
            try:
                from huggingface_hub import login, whoami
                
                # Try to login
                login(token=token)
                
                # Verify login
                user_info = whoami(token=token)
                
                self.hf_token = token
                self.hf_logged_in = True
                
                return jsonify({
                    'success': True,
                    'username': user_info.get('name', 'Unknown'),
                    'message': 'Successfully logged in to HuggingFace'
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 401
        
        @self.app.route('/api/hf/status')
        def hf_status():
            """Check HuggingFace login status."""
            return jsonify({
                'logged_in': self.hf_logged_in,
                'username': None  # Could be extended to store username
            })
        
        @self.app.route('/api/hf/datasets')
        def list_hf_datasets():
            """List popular HuggingFace datasets."""
            # Popular datasets for training
            popular_datasets = [
                {
                    'name': 'squad',
                    'full_name': 'rajpurkar/squad',
                    'description': 'Stanford Question Answering Dataset',
                    'size': '100K+ QA pairs',
                    'category': 'QA'
                },
                {
                    'name': 'wikitext',
                    'full_name': 'wikitext',
                    'description': 'Wikipedia text corpus',
                    'size': '100M+ tokens',
                    'category': 'Text'
                },
                {
                    'name': 'common_gen',
                    'full_name': 'common_gen',
                    'description': 'CommonGen dataset',
                    'size': '35K+ samples',
                    'category': 'Generation'
                },
                {
                    'name': 'eli5',
                    'full_name': 'eli5',
                    'description': 'Explain Like I\'m 5',
                    'size': '270K+ QA pairs',
                    'category': 'QA'
                },
                {
                    'name': 'natural_questions',
                    'full_name': 'natural_questions',
                    'description': 'Google Natural Questions',
                    'size': '300K+ questions',
                    'category': 'QA'
                },
                {
                    'name': 'trivia_qa',
                    'full_name': 'trivia_qa',
                    'description': 'Trivia Question Answering',
                    'size': '95K+ QA pairs',
                    'category': 'QA'
                }
            ]
            
            return jsonify({
                'datasets': popular_datasets,
                'requires_login': not self.hf_logged_in
            })
        
        @self.app.route('/api/hf/search')
        def search_hf_datasets():
            """Search HuggingFace datasets."""
            query = request.args.get('query', '')
            limit = int(request.args.get('limit', 20))
            
            if not query:
                return jsonify({'datasets': [], 'error': 'No query provided'}), 400
            
            try:
                from huggingface_hub import list_datasets
                
                # Search datasets
                datasets = list_datasets(
                    search=query,
                    limit=limit,
                    sort='downloads',
                    direction=-1
                )
                
                results = []
                for ds in datasets:
                    results.append({
                        'name': ds.id.split('/')[-1] if '/' in ds.id else ds.id,
                        'full_name': ds.id,
                        'description': getattr(ds, 'description', 'No description') or 'No description',
                        'downloads': getattr(ds, 'downloads', 0) or 0,
                        'likes': getattr(ds, 'likes', 0) or 0,
                        'tags': getattr(ds, 'tags', []) or []
                    })
                
                return jsonify({
                    'datasets': results,
                    'count': len(results),
                    'query': query
                })
                
            except Exception as e:
                return jsonify({'error': str(e), 'datasets': []}), 500
        
        @self.app.route('/api/datasets')
        def list_datasets():
            """List available training datasets (local + HF)."""
            from neuron_system.ai.datasets import DatasetLoader
            
            loader = DatasetLoader()
            local_datasets = loader.list_available_datasets()
            
            datasets = [
                {
                    'name': ds['name'],
                    'description': ds['description'],
                    'size': ds.get('size', 'Unknown'),
                    'source': 'local'
                }
                for ds in local_datasets
            ]
            
            return jsonify({
                'datasets': datasets,
                'hf_available': self.hf_logged_in
            })
        
        @self.app.route('/api/train', methods=['POST'])
        def train():
            """Train the brain with a dataset."""
            data = request.json
            dataset_name = data.get('dataset', 'conversations')
            max_samples = int(data.get('max_samples', 1000))
            source = data.get('source', 'local')
            
            try:
                if source == 'huggingface':
                    # Train with HuggingFace dataset
                    if not self.hf_logged_in:
                        return jsonify({'error': 'Please login to HuggingFace first'}), 401
                    
                    return self._train_huggingface(dataset_name, max_samples)
                else:
                    # Train with local dataset
                    return self._train_local(dataset_name, max_samples)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        def _train_local(self, dataset_name, max_samples):
            """Train with local dataset."""
            from neuron_system.ai.datasets import DatasetLoader
            
            loader = DatasetLoader()
            
            # Load dataset
            if dataset_name == 'conversations':
                dataset = loader.load_conversations(max_samples=max_samples)
            else:
                return jsonify({'error': f'Unknown dataset: {dataset_name}'}), 400
            
            # Start training in background thread
            def train_worker():
                self.training_active = True
                trained = 0
                errors = 0
                
                for i, item in enumerate(dataset):
                    try:
                        if 'question' in item and 'answer' in item:
                            # Get neurons before
                            neurons_before = len(self.graph.neurons)
                            
                            # Train
                            self.language_model.learn(
                                f"Q: {item['question']}\nA: {item['answer']}"
                            )
                            
                            # Get neurons after
                            neurons_after = len(self.graph.neurons)
                            
                            # If new neurons were added, broadcast them
                            if neurons_after > neurons_before:
                                new_neurons = list(self.graph.neurons.values())[-1:]
                                for neuron in new_neurons:
                                    self._broadcast_new_neuron(neuron)
                            
                            trained += 1
                            
                            # Broadcast progress every 10 samples
                            if trained % 10 == 0:
                                self.broadcast_training_progress({
                                    'trained': trained,
                                    'total': max_samples,
                                    'errors': errors,
                                    'total_neurons': len(self.graph.neurons)
                                })
                    except Exception as e:
                        errors += 1
                
                # Save
                self.graph.save()
                self.training_active = False
                
                # Final progress
                self.broadcast_training_progress({
                    'trained': trained,
                    'total': max_samples,
                    'errors': errors,
                    'total_neurons': len(self.graph.neurons),
                    'complete': True
                })
            
            # Start training thread
            self.training_thread = threading.Thread(target=train_worker)
            self.training_thread.start()
            
            return jsonify({
                'success': True,
                'message': 'Training started',
                'live_updates': True
            })
        
        def _train_huggingface(self, dataset_name, max_samples):
            """Train with HuggingFace dataset."""
            from datasets import load_dataset
            
            def train_worker():
                self.training_active = True
                trained = 0
                errors = 0
                
                try:
                    print(f"Loading HuggingFace dataset: {dataset_name}")
                    
                    # Load dataset from HuggingFace
                    dataset = load_dataset(dataset_name, split='train', streaming=True)
                    
                    # Process samples
                    for i, item in enumerate(dataset):
                        if i >= max_samples:
                            break
                        
                        try:
                            # Extract text based on dataset structure
                            text = self._extract_text_from_hf_item(item, dataset_name)
                            
                            if text:
                                # Get neurons before
                                neurons_before = len(self.graph.neurons)
                                
                                # Train
                                self.language_model.learn(text)
                                
                                # Get neurons after
                                neurons_after = len(self.graph.neurons)
                                
                                # If new neurons were added, broadcast them
                                if neurons_after > neurons_before:
                                    new_neurons = list(self.graph.neurons.values())[-1:]
                                    for neuron in new_neurons:
                                        self._broadcast_new_neuron(neuron)
                                
                                trained += 1
                                
                                # Broadcast progress every 10 samples
                                if trained % 10 == 0:
                                    self.broadcast_training_progress({
                                        'trained': trained,
                                        'total': max_samples,
                                        'errors': errors,
                                        'total_neurons': len(self.graph.neurons)
                                    })
                        except Exception as e:
                            errors += 1
                            print(f"Error processing item {i}: {e}")
                    
                    # Save
                    self.graph.save()
                    
                    # Final progress
                    self.broadcast_training_progress({
                        'trained': trained,
                        'total': max_samples,
                        'errors': errors,
                        'total_neurons': len(self.graph.neurons),
                        'complete': True
                    })
                    
                except Exception as e:
                    self.broadcast_training_progress({
                        'error': str(e),
                        'complete': True
                    })
                finally:
                    self.training_active = False
            
            # Start training thread
            self.training_thread = threading.Thread(target=train_worker)
            self.training_thread.start()
            
            return jsonify({
                'success': True,
                'message': 'Training started',
                'live_updates': True
            })
        
        def _extract_text_from_hf_item(self, item, dataset_name):
            """Extract text from HuggingFace dataset item."""
            # Handle different dataset structures
            if 'question' in item and 'answers' in item:
                # SQuAD-like format
                question = item['question']
                answers = item['answers']
                if isinstance(answers, dict) and 'text' in answers:
                    answer = answers['text'][0] if answers['text'] else ''
                else:
                    answer = str(answers)
                return f"Q: {question}\nA: {answer}"
            
            elif 'question' in item and 'answer' in item:
                # Simple QA format
                return f"Q: {item['question']}\nA: {item['answer']}"
            
            elif 'text' in item:
                # Text corpus format
                return item['text']
            
            elif 'content' in item:
                return item['content']
            
            elif 'document' in item:
                return item['document']
            
            else:
                # Try to convert entire item to string
                return str(item)
        
        def _broadcast_new_neuron(self, neuron):
            """Broadcast a new neuron to all clients."""
            try:
                n_dict = neuron.to_dict()
                pos = n_dict.get('position', {})
                
                neuron_data = {
                    'id': str(neuron.id),
                    'position': {
                        'x': pos.get('x', 0),
                        'y': pos.get('y', 0),
                        'z': pos.get('z', 0)
                    },
                    'neuron_type': n_dict.get('neuron_type', 'KNOWLEDGE')
                }
                
                self.broadcast_neuron_added(neuron_data)
            except Exception as e:
                print(f"Error broadcasting neuron: {e}")
        
        @self.app.route('/api/learn', methods=['POST'])
        def learn():
            """Add single knowledge item."""
            data = request.json
            text = data.get('text', '')
            tags = data.get('tags', [])
            
            if not text:
                return jsonify({'error': 'No text provided'}), 400
            
            try:
                self.language_model.learn(text, tags=tags)
                self.graph.save()
                
                return jsonify({
                    'success': True,
                    'total_neurons': len(self.graph.neurons)
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stats')
        def get_stats_api():
            """Get detailed statistics."""
            stats = self.get_stats()
            
            # Get neuron types
            neuron_types = {}
            for neuron in self.graph.neurons.values():
                ntype = getattr(neuron, 'neuron_type', 'KNOWLEDGE')
                neuron_types[ntype] = neuron_types.get(ntype, 0) + 1
            
            stats['neuron_types'] = neuron_types
            
            return jsonify(stats)
    
    def get_stats(self):
        """Get brain statistics."""
        neurons = self.neuron_store.list_all()
        
        with self.synapse_store.db.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM synapses")
            total_synapses = cursor.fetchone()[0]
        
        return {
            'neurons': len(neurons),
            'synapses': total_synapses
        }
    
    def _setup_socketio(self):
        """Setup SocketIO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
            emit('connected', {'message': 'Connected to brain viewer'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Client disconnected')
    
    def broadcast_neuron_added(self, neuron_data):
        """Broadcast new neuron to all connected clients."""
        if self.training_active:
            self.socketio.emit('neuron_added', neuron_data)
    
    def broadcast_synapse_added(self, synapse_data):
        """Broadcast new synapse to all connected clients."""
        if self.training_active:
            self.socketio.emit('synapse_added', synapse_data)
    
    def broadcast_training_progress(self, progress_data):
        """Broadcast training progress."""
        self.socketio.emit('training_progress', progress_data)
    
    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Start the web server."""
        stats = self.get_stats()
        
        print(f"""
================================================================
        HIGH-PERFORMANCE 3D BRAIN VIEWER (Three.js)
================================================================

Database: {self.database_path}

Current State:
   Neurons:        {stats['neurons']}
   Total Synapses: {stats['synapses']:,}

Features:
   - WebGL-accelerated rendering
   - Can display 100K+ synapses smoothly
   - Real-time filtering
   - Interactive controls
   - Live training updates (WebSocket)

Starting server on http://localhost:{port}
Press Ctrl+C to stop
================================================================
""")
        
        self.socketio.run(self.app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="High-performance 3D brain viewer")
    parser.add_argument(
        '--database',
        default='data/neuron_system.db',
        help='Database path'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5001,
        help='Port for web server (default: 5001)'
    )
    
    args = parser.parse_args()
    
    viewer = ThreeJSBrainViewer(args.database)
    viewer.run(port=args.port)


if __name__ == "__main__":
    main()
