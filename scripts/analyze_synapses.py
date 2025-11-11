"""
Analyze Synapse Distribution - See weight distribution and connection patterns.
"""

import sqlite3
import numpy as np
from collections import Counter


def main():
    conn = sqlite3.connect('data/neuron_system.db')
    cursor = conn.cursor()
    
    # Get all synapses
    cursor.execute("SELECT weight, synapse_type FROM synapses")
    synapses = cursor.fetchall()
    
    if not synapses:
        print("No synapses found!")
        return
    
    weights = [s[0] for s in synapses]
    types = [s[1] for s in synapses]
    
    # Statistics
    weights_array = np.array(weights)
    
    print(f"""
================================================================
                SYNAPSE ANALYSIS
================================================================

Total Synapses: {len(synapses):,}

Weight Distribution:
   Min:        {weights_array.min():.4f}
   Max:        {weights_array.max():.4f}
   Mean:       {weights_array.mean():.4f}
   Median:     {np.median(weights_array):.4f}
   Std Dev:    {weights_array.std():.4f}

Weight Ranges:
   Very Strong (>0.8):   {np.sum(np.abs(weights_array) > 0.8):,} ({np.sum(np.abs(weights_array) > 0.8)/len(weights)*100:.1f}%)
   Strong (0.5-0.8):     {np.sum((np.abs(weights_array) > 0.5) & (np.abs(weights_array) <= 0.8)):,} ({np.sum((np.abs(weights_array) > 0.5) & (np.abs(weights_array) <= 0.8))/len(weights)*100:.1f}%)
   Medium (0.3-0.5):     {np.sum((np.abs(weights_array) > 0.3) & (np.abs(weights_array) <= 0.5)):,} ({np.sum((np.abs(weights_array) > 0.3) & (np.abs(weights_array) <= 0.5))/len(weights)*100:.1f}%)
   Weak (<0.3):          {np.sum(np.abs(weights_array) <= 0.3):,} ({np.sum(np.abs(weights_array) <= 0.3)/len(weights)*100:.1f}%)

Synapse Types:
""")
    
    type_counts = Counter(types)
    for stype, count in type_counts.most_common():
        print(f"   {stype}: {count:,} ({count/len(types)*100:.1f}%)")
    
    # Recommendations
    print(f"""
================================================================
                VISUALIZATION RECOMMENDATIONS
================================================================

For best performance in 3D viewer:

1. Show only strong connections:
   python view_brain_optimized.py --min-weight 0.5 --max-synapses 500
   (Shows ~{np.sum(np.abs(weights_array) > 0.5):,} synapses)

2. Show very strong connections only:
   python view_brain_optimized.py --min-weight 0.8 --max-synapses 1000
   (Shows ~{np.sum(np.abs(weights_array) > 0.8):,} synapses)

3. Show medium+ connections:
   python view_brain_optimized.py --min-weight 0.3 --max-synapses 2000
   (Shows ~{np.sum(np.abs(weights_array) > 0.3):,} synapses)

================================================================
""")
    
    conn.close()


if __name__ == "__main__":
    main()
