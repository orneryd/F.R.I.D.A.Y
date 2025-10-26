"""
Count total conversation items.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuron_system.ai.conversation_knowledge import ConversationKnowledge
from neuron_system.ai.natural_dialogue import NaturalDialogue

# Count items
conv_items = ConversationKnowledge.get_all_knowledge()
dialogue_items = NaturalDialogue.get_all_knowledge()

print("=" * 70)
print("CONVERSATION DATA COUNT")
print("=" * 70)
print()
print(f"ConversationKnowledge items: {len(conv_items)}")
print(f"NaturalDialogue items: {len(dialogue_items)}")
print(f"Total: {len(conv_items) + len(dialogue_items)}")
print()

# Count by category
print("ConversationKnowledge categories:")
categories = {}
for item in conv_items:
    for tag in item['tags']:
        categories[tag] = categories.get(tag, 0) + 1

for cat, count in sorted(categories.items()):
    print(f"  {cat}: {count}")

print()
print("NaturalDialogue categories:")
categories = {}
for item in dialogue_items:
    for tag in item['tags']:
        categories[tag] = categories.get(tag, 0) + 1

for cat, count in sorted(categories.items()):
    print(f"  {cat}: {count}")
