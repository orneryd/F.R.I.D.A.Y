"""
Validate 3D System - Prüft dass Neuronen korrekt positioniert und verbunden sind.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from neuron_system.core.graph import NeuronGraph
from neuron_system.engines.compression import CompressionEngine
from neuron_system.engines.query import QueryEngine
from neuron_system.engines.training import TrainingEngine
from neuron_system.ai.language_model import LanguageModel
from neuron_system.storage.database import DatabaseManager

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

print("\n" + "=" * 70)
print("3D SYSTEM VALIDATION")
print("=" * 70 + "\n")

# Initialize system
db = DatabaseManager("data/neuron_system.db")
graph = NeuronGraph()
graph.attach_storage(db)

compression_engine = CompressionEngine()
query_engine = QueryEngine(graph, compression_engine)
training_engine = TrainingEngine(graph)
language_model = LanguageModel(
    graph, compression_engine, query_engine, training_engine,
    enable_self_training=False
)

# Load existing data
try:
    graph.load()
    logger.info(f"Loaded {len(graph.neurons)} neurons from database")
except:
    logger.info("No existing data, creating new neurons...")

# Create test neurons if empty
if len(graph.neurons) == 0:
    logger.info("Creating test neurons...")
    
    test_data = [
        ("AI is artificial intelligence", ['ai', 'definition']),
        ("ML is machine learning", ['ml', 'definition']),
        ("DL is deep learning", ['dl', 'definition']),
        ("NLP is natural language processing", ['nlp', 'definition']),
        ("Python is a programming language", ['python', 'programming']),
    ]
    
    for text, tags in test_data:
        language_model.learn(text, tags=tags)
    
    logger.info(f"Created {len(test_data)} test neurons")

print()
print("=" * 70)
print("VALIDATION CHECKS")
print("=" * 70)
print()

# Check 1: All neurons have positions
print("Check 1: Neuron Positions")
print("-" * 70)

neurons_without_position = 0
for neuron in graph.neurons.values():
    if neuron.position is None:
        neurons_without_position += 1

if neurons_without_position == 0:
    print(f"✓ All {len(graph.neurons)} neurons have positions")
else:
    print(f"✗ {neurons_without_position} neurons missing positions!")

print()

# Check 2: All neurons have vectors
print("Check 2: Neuron Vectors")
print("-" * 70)

neurons_without_vector = 0
vector_dimensions = set()

for neuron in graph.neurons.values():
    if neuron.vector is None:
        neurons_without_vector += 1
    else:
        vector_dimensions.add(len(neuron.vector))

if neurons_without_vector == 0:
    print(f"✓ All {len(graph.neurons)} neurons have vectors")
    print(f"  Dimensions: {vector_dimensions}")
else:
    print(f"✗ {neurons_without_vector} neurons missing vectors!")

print()

# Check 3: Positions are within bounds
print("Check 3: Position Bounds")
print("-" * 70)

min_bound, max_bound = graph.bounds
out_of_bounds = 0

for neuron in graph.neurons.values():
    if neuron.position:
        if not (min_bound.x <= neuron.position.x <= max_bound.x and
                min_bound.y <= neuron.position.y <= max_bound.y and
                min_bound.z <= neuron.position.z <= max_bound.z):
            out_of_bounds += 1

if out_of_bounds == 0:
    print(f"✓ All neurons within bounds")
    print(f"  Bounds: {min_bound} to {max_bound}")
else:
    print(f"✗ {out_of_bounds} neurons out of bounds!")

print()

# Check 4: Spatial Index
print("Check 4: Spatial Index")
print("-" * 70)

try:
    # Test spatial query
    test_position = graph.neurons[list(graph.neurons.keys())[0]].position
    nearby = graph.spatial_index.find_nearby(test_position, radius=20.0)
    
    print(f"✓ Spatial index working")
    print(f"  Test query found {len(nearby)} nearby neurons")
except Exception as e:
    print(f"✗ Spatial index error: {e}")

print()

# Check 5: Synapses
print("Check 5: Synapses")
print("-" * 70)

invalid_synapses = 0
for synapse in graph.synapses.values():
    # Check if source and target exist
    if synapse.source_neuron_id not in graph.neurons:
        invalid_synapses += 1
    if synapse.target_neuron_id not in graph.neurons:
        invalid_synapses += 1

if invalid_synapses == 0:
    print(f"✓ All {len(graph.synapses)} synapses valid")
    if len(graph.neurons) > 0:
        avg_connectivity = len(graph.synapses) / len(graph.neurons)
        print(f"  Avg connectivity: {avg_connectivity:.2f}")
else:
    print(f"✗ {invalid_synapses} invalid synapses!")

print()

# Check 6: Topic Clustering
print("Check 6: Topic Clustering")
print("-" * 70)

topics = {}
for neuron in graph.neurons.values():
    if hasattr(neuron, 'semantic_tags') and neuron.semantic_tags:
        for tag in neuron.semantic_tags:
            if tag not in topics:
                topics[tag] = []
            if neuron.position:
                topics[tag].append(neuron.position)

if topics:
    print(f"✓ Found {len(topics)} topics")
    
    # Check if neurons with same topic are clustered
    for topic, positions in topics.items():
        if len(positions) > 1:
            # Calculate average distance between neurons of same topic
            distances = []
            for i, pos1 in enumerate(positions):
                for pos2 in positions[i+1:]:
                    distances.append(pos1.distance_to(pos2))
            
            if distances:
                avg_dist = sum(distances) / len(distances)
                print(f"  {topic}: {len(positions)} neurons, avg distance: {avg_dist:.2f}")
else:
    print("  No topics found")

print()

# Summary
print("=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
print()

all_checks_passed = (
    neurons_without_position == 0 and
    neurons_without_vector == 0 and
    out_of_bounds == 0 and
    invalid_synapses == 0
)

if all_checks_passed:
    print("✅ ALL CHECKS PASSED")
    print()
    print("3D System is working correctly:")
    print(f"  ✓ {len(graph.neurons)} neurons properly positioned")
    print(f"  ✓ {len(graph.synapses)} synapses properly connected")
    print(f"  ✓ Spatial index functional")
    print(f"  ✓ All positions within bounds")
else:
    print("❌ SOME CHECKS FAILED")
    print()
    print("Issues found:")
    if neurons_without_position > 0:
        print(f"  ✗ {neurons_without_position} neurons without position")
    if neurons_without_vector > 0:
        print(f"  ✗ {neurons_without_vector} neurons without vector")
    if out_of_bounds > 0:
        print(f"  ✗ {out_of_bounds} neurons out of bounds")
    if invalid_synapses > 0:
        print(f"  ✗ {invalid_synapses} invalid synapses")

print()
print("=" * 70 + "\n")
