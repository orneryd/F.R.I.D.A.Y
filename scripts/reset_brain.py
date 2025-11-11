"""
Reset the brain - delete all neurons and synapses
"""
import os
from pathlib import Path
from neuron_system.storage.database import DatabaseManager

def reset_brain():
    """Delete the entire database and start fresh"""
    db_path = Path("data/neuron_system.db")
    
    if db_path.exists():
        print(f"🗑️  Deleting existing brain database: {db_path}")
        os.remove(db_path)
        print("✅ Brain reset complete!")
    else:
        print("ℹ️  No existing brain found - starting fresh")
    
    # Create new empty database
    print("🧠 Creating new empty brain...")
    db_manager = DatabaseManager(str(db_path))
    print("✅ New brain initialized!")
    print()
    print("Ready to train. Run:")
    print("  python assimilate.py")

if __name__ == "__main__":
    print("=" * 60)
    print("FRIDAY AI - BRAIN RESET")
    print("=" * 60)
    print()
    
    response = input("⚠️  This will DELETE all neurons and synapses. Continue? (yes/no): ")
    
    if response.lower() == 'yes':
        reset_brain()
    else:
        print("❌ Reset cancelled")
