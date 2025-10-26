# F.R.I.D.A.Y Tests & Utilities

This directory contains test scripts, utilities, and migration tools for the F.R.I.D.A.Y AI system.

## Test Categories

### Training Tests
- `test_aggressive_training.py` - Test aggressive self-training system
- `test_training_improvement.py` - Test training improvement over multiple rounds
- `test_self_training.py` - Basic self-training tests
- `train_comprehensive.py` - Comprehensive training script
- `train_incremental.py` - Incremental training script

### AI Response Tests
- `test_improved_ai.py` - Test improved AI responses
- `test_ai_quick.py` - Quick AI response tests
- `test_ai_responses.py` - Comprehensive AI response tests
- `test_problem_questions.py` - Test problematic questions
- `test_what_are_you.py` - Test identity questions

### Reasoning & Reflection Tests
- `test_reasoning.py` - Test reasoning capabilities
- `test_self_reflection.py` - Test self-reflection system
- `test_complex_question.py` - Test complex question handling

### Conversation Tests
- `test_conversation.py` - Basic conversation tests
- `test_conversation_detailed.py` - Detailed conversation tests

### Query Engine Tests
- `test_query_engine.py` - Test query engine
- `test_query_detailed.py` - Detailed query tests
- `test_language_query.py` - Language query tests

### Specific Feature Tests
- `test_letter_counting.py` - Test letter counting (e.g., "strawberry")
- `test_strawberry.py` - Strawberry letter counting test

### Storage & Database Tests
- `test_storage.py` - Test storage system
- `test_save.py` - Test save functionality
- `check_db.py` - Database inspection utility

### Utilities
- `count_conversations.py` - Count conversation neurons
- `count_reasoning.py` - Count reasoning neurons
- `debug_vectors.py` - Debug vector embeddings
- `migrate_add_importance.py` - Migration: Add importance attribute to neurons

## Running Tests

### Individual Tests
```bash
# Run a specific test
python tests/test_aggressive_training.py

# Run training improvement test
python tests/test_training_improvement.py
```

### Using CLI (Recommended)
```bash
# Run self-learning training
python cli.py learn --rounds 3 --save

# Test AI with predefined questions
python cli.py test

# Show statistics
python cli.py stats
```

## Test Databases

Some tests create their own databases:
- `test_save.db` - Test save functionality
- `test_storage.db` - Test storage system

These are temporary and can be deleted.

## Notes

- Most tests use `comprehensive_ai.db` as the main database
- Tests with `--save` flag will persist changes to the database
- Migration scripts should be run once when updating the system
- Utility scripts are for debugging and inspection only
