"""
Count reasoning items.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuron_system.ai.reasoning_knowledge import ReasoningKnowledge

# Count items
reasoning_items = ReasoningKnowledge.get_all_knowledge()

print("=" * 70)
print("REASONING DATA COUNT")
print("=" * 70)
print()
print(f"Total Reasoning Items: {len(reasoning_items)}")
print()

# Count by category
categories = {}
for item in reasoning_items:
    for tag in item['tags']:
        categories[tag] = categories.get(tag, 0) + 1

print("Categories:")
for cat, count in sorted(categories.items()):
    print(f"  {cat}: {count}")
