"""
Extract Knowledge from Qwen3 into Friday's Brain.

This tool extracts intelligence from the Qwen3 model and stores it
as neurons, making Friday independent of external models.
"""

import argparse
import logging
from neuron_system.training import TrainingManager
from neuron_system.training.knowledge_extractor import KnowledgeExtractor
from neuron_system.training.logic_extractor import LogicExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text: str):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def extract_knowledge(args):
    """Extract knowledge from Qwen."""
    print_header("KNOWLEDGE EXTRACTION FROM QWEN3")
    
    print("\n🧠 This will extract Qwen3's intelligence into Friday's neurons")
    print("   After extraction, Friday won't need external models!")
    
    # Initialize
    print("\nInitializing...")
    manager = TrainingManager(args.database)
    extractor = KnowledgeExtractor(manager)
    
    print(f"✓ Current neurons: {len(manager.graph.neurons)}")
    
    # Extract
    print_header("EXTRACTING KNOWLEDGE")
    
    stats = extractor.extract_from_qwen(
        topics=None,  # Use default topics
        questions_per_topic=args.questions_per_topic
    )
    
    if "error" in stats:
        print(f"\n❌ Error: {stats['error']}")
        return
    
    print_header("EXTRACTION COMPLETE")
    print(f"\n✅ Extracted: {stats['extracted']} knowledge pieces")
    print(f"   Topics: {stats['topics']}")
    print(f"   Total neurons: {stats['total_neurons']}")
    
    print("\n🎉 Friday's brain now contains Qwen3's knowledge!")
    print("   Friday can now answer questions without external models.")


def extract_patterns(args):
    """Extract language patterns."""
    print_header("LANGUAGE PATTERN EXTRACTION")
    
    manager = TrainingManager(args.database)
    extractor = KnowledgeExtractor(manager)
    
    stats = extractor.extract_language_patterns(
        num_patterns=args.num_patterns
    )
    
    if "error" in stats:
        print(f"\n❌ Error: {stats['error']}")
        return
    
    print_header("EXTRACTION COMPLETE")
    print(f"\n✅ Extracted: {stats['extracted']} patterns")
    print(f"   Total neurons: {stats['total_neurons']}")


def extract_reasoning(args):
    """Extract reasoning patterns."""
    print_header("REASONING PATTERN EXTRACTION")
    
    manager = TrainingManager(args.database)
    extractor = KnowledgeExtractor(manager)
    
    stats = extractor.extract_reasoning_patterns(
        num_examples=args.num_examples
    )
    
    if "error" in stats:
        print(f"\n❌ Error: {stats['error']}")
        return
    
    print_header("EXTRACTION COMPLETE")
    print(f"\n✅ Extracted: {stats['extracted']} reasoning patterns")
    print(f"   Total neurons: {stats['total_neurons']}")


def extract_logic(args):
    """Extract thinking logic from Qwen."""
    print_header("LOGIC EXTRACTION - HOW QWEN THINKS")
    
    print("\n🧠 Extracting Qwen's THINKING LOGIC...")
    print("   This extracts HOW Qwen thinks, not just WHAT it knows!")
    print("   - Attention patterns (what to focus on)")
    print("   - Generation strategy (how to choose words)")
    print("   - Reasoning logic (how to think step-by-step)")
    print("   - Composition patterns (how to structure responses)")
    
    manager = TrainingManager(args.database)
    logic_extractor = LogicExtractor(manager)
    
    initial_neurons = len(manager.graph.neurons)
    
    # Extract attention patterns
    print("\n1️⃣ Extracting attention patterns...")
    stats1 = logic_extractor.extract_attention_patterns(num_examples=30)
    
    # Extract generation logic
    print("\n2️⃣ Extracting generation logic...")
    stats2 = logic_extractor.extract_generation_logic(num_examples=30)
    
    # Extract reasoning logic
    print("\n3️⃣ Extracting reasoning logic...")
    stats3 = logic_extractor.extract_reasoning_logic(num_examples=20)
    
    # Extract composition patterns
    print("\n4️⃣ Extracting composition patterns...")
    stats4 = logic_extractor.extract_composition_patterns(num_examples=30)
    
    final_neurons = len(manager.graph.neurons)
    new_neurons = final_neurons - initial_neurons
    
    print_header("LOGIC EXTRACTION COMPLETE")
    print(f"\n✅ Total logic neurons: {new_neurons}")
    print(f"   Attention: {stats1.get('extracted', 0)}")
    print(f"   Generation: {stats2.get('extracted', 0)}")
    print(f"   Reasoning: {stats3.get('extracted', 0)}")
    print(f"   Composition: {stats4.get('extracted', 0)}")
    print(f"\n   Total neurons: {final_neurons}")
    
    print("\n🎉 Friday now has Qwen's THINKING LOGIC!")
    print("   Friday is DYNAMIC, not static!")


def extract_all(args):
    """Extract everything - knowledge AND logic."""
    print_header("FULL EXTRACTION - KNOWLEDGE + LOGIC")
    
    print("\n🧠 Extracting EVERYTHING from Qwen3...")
    print("   This will take 10-15 minutes.")
    print("\n   Phase 1: Knowledge (what Qwen knows)")
    print("   Phase 2: Logic (how Qwen thinks)")
    
    manager = TrainingManager(args.database)
    knowledge_extractor = KnowledgeExtractor(manager)
    logic_extractor = LogicExtractor(manager)
    
    initial_neurons = len(manager.graph.neurons)
    
    # Phase 1: Logic (FIRST!)
    print_header("PHASE 1: LOGIC EXTRACTION")
    print("\n⚡ Extracting HOW Qwen thinks (logic defines thinking)...")
    
    print("\n1️⃣ Extracting attention patterns...")
    stats1 = logic_extractor.extract_attention_patterns(num_examples=30)
    
    print("\n2️⃣ Extracting generation logic...")
    stats2 = logic_extractor.extract_generation_logic(num_examples=30)
    
    print("\n3️⃣ Extracting reasoning logic...")
    stats3 = logic_extractor.extract_reasoning_logic(num_examples=20)
    
    print("\n4️⃣ Extracting composition patterns...")
    stats4 = logic_extractor.extract_composition_patterns(num_examples=30)
    
    # Phase 2: Knowledge (AFTER logic!)
    print_header("PHASE 2: KNOWLEDGE EXTRACTION")
    print("\n📡 Extracting WHAT Qwen knows (processed with logic)...")
    
    print("\n5️⃣ Extracting knowledge...")
    stats5 = knowledge_extractor.extract_from_qwen(questions_per_topic=10)
    
    print("\n6️⃣ Extracting language patterns...")
    stats6 = knowledge_extractor.extract_language_patterns(num_patterns=100)
    
    print("\n7️⃣ Extracting reasoning patterns...")
    stats7 = knowledge_extractor.extract_reasoning_patterns(num_examples=50)
    
    final_neurons = len(manager.graph.neurons)
    new_neurons = final_neurons - initial_neurons
    
    print_header("FULL EXTRACTION COMPLETE")
    print(f"\n✅ Total extracted: {new_neurons} neurons")
    print(f"\n   Logic Phase (extracted FIRST):")
    print(f"     - Attention: {stats1.get('extracted', 0)}")
    print(f"     - Generation: {stats2.get('extracted', 0)}")
    print(f"     - Reasoning Logic: {stats3.get('extracted', 0)}")
    print(f"     - Composition: {stats4.get('extracted', 0)}")
    print(f"\n   Knowledge Phase (processed with logic):")
    print(f"     - Knowledge: {stats5.get('extracted', 0)}")
    print(f"     - Patterns: {stats6.get('extracted', 0)}")
    print(f"     - Reasoning: {stats7.get('extracted', 0)}")
    print(f"\n   Total neurons: {final_neurons}")
    
    print("\n🎉 Friday is now TRULY INTELLIGENT!")
    print("   ⚡ Logic extracted FIRST (defines HOW to think)")
    print("   📡 Knowledge extracted AFTER (processed with logic)")
    print("   🧠 Is DYNAMIC, not static")
    print("   🚀 No external models needed!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract knowledge from Qwen3 into Friday's brain"
    )
    
    parser.add_argument(
        '--database',
        default='data/neuron_system.db',
        help='Database path'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Extraction command')
    
    # Knowledge extraction
    knowledge_parser = subparsers.add_parser(
        'knowledge',
        help='Extract knowledge about topics'
    )
    knowledge_parser.add_argument(
        '--questions-per-topic',
        type=int,
        default=10,
        help='Questions per topic (default: 10)'
    )
    
    # Pattern extraction
    pattern_parser = subparsers.add_parser(
        'patterns',
        help='Extract language patterns'
    )
    pattern_parser.add_argument(
        '--num-patterns',
        type=int,
        default=100,
        help='Number of patterns (default: 100)'
    )
    
    # Reasoning extraction
    reasoning_parser = subparsers.add_parser(
        'reasoning',
        help='Extract reasoning patterns'
    )
    reasoning_parser.add_argument(
        '--num-examples',
        type=int,
        default=50,
        help='Number of examples (default: 50)'
    )
    
    # Logic extraction
    subparsers.add_parser(
        'logic',
        help='Extract thinking logic (HOW Qwen thinks)'
    )
    
    # Full extraction
    subparsers.add_parser(
        'all',
        help='Extract everything (knowledge + logic)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'knowledge':
        extract_knowledge(args)
    elif args.command == 'patterns':
        extract_patterns(args)
    elif args.command == 'reasoning':
        extract_reasoning(args)
    elif args.command == 'logic':
        extract_logic(args)
    elif args.command == 'all':
        extract_all(args)


if __name__ == "__main__":
    main()
