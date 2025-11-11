"""
Unified Training CLI for Friday.

Single entry point for all training operations.
"""

import argparse
import logging
from pathlib import Path

from neuron_system.training import TrainingManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def train_conversations(args):
    """Train from conversation dataset."""
    print_header("TRAINING: CONVERSATIONS")
    
    manager = TrainingManager(args.database)
    
    stats = manager.train_conversations(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    print_header("TRAINING COMPLETE")
    print(f"\nStatistics:")
    print(f"  Trained: {stats['trained']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Errors: {stats.get('errors', 0)}")
    print(f"  Total neurons: {stats['total_neurons']}")


def train_qa(args):
    """Train from Q&A file."""
    print_header("TRAINING: Q&A PAIRS")
    
    # Load Q&A data from file
    import json
    
    qa_file = Path(args.file)
    if not qa_file.exists():
        logger.error(f"File not found: {args.file}")
        return
    
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    manager = TrainingManager(args.database)
    stats = manager.train_qa_pairs(qa_data)
    
    print_header("TRAINING COMPLETE")
    print(f"\nStatistics:")
    print(f"  Trained: {stats['trained']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Total neurons: {stats['total_neurons']}")


def show_stats(args):
    """Show training statistics."""
    print_header("TRAINING STATISTICS")
    
    manager = TrainingManager(args.database)
    stats = manager.get_statistics()
    
    print(f"\nDatabase: {stats['database_path']}")
    print(f"Total neurons: {stats['total_neurons']}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Friday Training System - Unified training CLI"
    )
    
    parser.add_argument(
        '--database',
        default='data/neuron_system.db',
        help='Database path (default: data/neuron_system.db)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Training command')
    
    # Conversations command
    conv_parser = subparsers.add_parser(
        'conversations',
        help='Train from conversation dataset'
    )
    conv_parser.add_argument(
        '--dataset',
        default='shihyunlim/english-conversation',
        help='HuggingFace dataset name'
    )
    conv_parser.add_argument(
        '--max-samples',
        type=int,
        default=3000,
        help='Maximum samples to train (default: 3000)'
    )
    conv_parser.add_argument(
        '--min-length',
        type=int,
        default=15,
        help='Minimum text length (default: 15)'
    )
    conv_parser.add_argument(
        '--max-length',
        type=int,
        default=1000,
        help='Maximum text length (default: 1000)'
    )
    
    # Q&A command
    qa_parser = subparsers.add_parser(
        'qa',
        help='Train from Q&A pairs file'
    )
    qa_parser.add_argument(
        'file',
        help='JSON file with Q&A pairs'
    )
    
    # Stats command
    subparsers.add_parser(
        'stats',
        help='Show training statistics'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'conversations':
        train_conversations(args)
    elif args.command == 'qa':
        train_qa(args)
    elif args.command == 'stats':
        show_stats(args)


if __name__ == "__main__":
    main()
