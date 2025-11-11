"""
Fix neurons that have no position (0,0,0)
Assigns random positions to neurons without valid positions
"""
import numpy as np
from neuron_system.storage.database import DatabaseManager
from neuron_system.storage.neuron_store import NeuronStore
from neuron_system.core.vector3d import Vector3D


def fix_positions():
    """Assign positions to neurons that don't have them"""
    print("=" * 60)
    print("FIX NEURON POSITIONS")
    print("=" * 60)
    print()
    
    # Initialize
    db = DatabaseManager('data/neuron_system.db')
    neuron_store = NeuronStore(db)
    
    # Get all neurons
    neurons = neuron_store.list_all()
    print(f"Total neurons: {len(neurons)}")
    
    # Find neurons without positions
    neurons_to_fix = []
    for neuron in neurons:
        neuron_dict = neuron.to_dict()
        pos = neuron_dict.get('position', {'x': 0, 'y': 0, 'z': 0})
        
        if pos.get('x', 0) == 0 and pos.get('y', 0) == 0 and pos.get('z', 0) == 0:
            neurons_to_fix.append(neuron)
    
    print(f"Neurons without position: {len(neurons_to_fix)}")
    print()
    
    if len(neurons_to_fix) == 0:
        print("✅ All neurons have positions!")
        return
    
    # Use brain-shaped positioning
    from neuron_system.visualization.brain_layout import generate_brain_shaped_position
    
    print("Assigning brain-shaped positions...")
    
    for i, neuron in enumerate(neurons_to_fix):
        # Generate position
        x, y, z = generate_brain_shaped_position(
            index=i,
            total=len(neurons_to_fix),
            scale=50.0
        )
        
        # Set position
        neuron.position = Vector3D(x=x, y=y, z=z)
        
        # Update in database
        neuron_store.update(neuron)
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(neurons_to_fix)}")
    
    print()
    print(f"✅ Fixed {len(neurons_to_fix)} neurons")
    print()
    print("Restart the live viewer to see the changes:")
    print("  python live_viewer.py")


if __name__ == "__main__":
    fix_positions()
