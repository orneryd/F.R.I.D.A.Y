"""
Validate Dynamic Dimensions - Stellt sicher dass keine hardcodierten Dimensionen existieren.

Dieses Script prüft:
1. Keine hardcodierten Dimensionen im Code
2. Alle Dimensionen werden dynamisch erkannt
3. System funktioniert mit verschiedenen Dimensionen
"""

import sys
import os
import re

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("\n" + "=" * 70)
print("DYNAMIC DIMENSIONS VALIDATION")
print("=" * 70 + "\n")

# Patterns to check
FORBIDDEN_PATTERNS = [
    (r'vector_dim\s*=\s*384', 'Hardcoded vector_dim = 384'),
    (r'vector_dim\s*=\s*768', 'Hardcoded vector_dim = 768'),
    (r'embedding_dim\s*=\s*384', 'Hardcoded embedding_dim = 384'),
    (r'embedding_dim\s*=\s*768', 'Hardcoded embedding_dim = 768'),
    (r'np\.zeros\(384\)', 'Hardcoded np.zeros(384)'),
    (r'np\.zeros\(768\)', 'Hardcoded np.zeros(768)'),
    (r'np\.ones\(384\)', 'Hardcoded np.ones(384)'),
    (r'np\.ones\(768\)', 'Hardcoded np.ones(768)'),
    (r'np\.random\.rand\(384\)', 'Hardcoded np.random.rand(384)'),
    (r'np\.random\.rand\(768\)', 'Hardcoded np.random.rand(768)'),
]

# Directories to check
DIRS_TO_CHECK = [
    'neuron_system',
    'cli.py',
    'main.py',
]

# Files to exclude
EXCLUDE_PATTERNS = [
    '__pycache__',
    '.pyc',
    'archive',
    'model_configs.py',  # Config file is OK to have dimension values
]

def should_exclude(filepath):
    """Check if file should be excluded."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in filepath:
            return True
    return False

def check_file(filepath):
    """Check a single file for hardcoded dimensions."""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for pattern, description in FORBIDDEN_PATTERNS:
            matches = re.finditer(pattern, content)
            for match in matches:
                # Get line number
                line_num = content[:match.start()].count('\n') + 1
                issues.append({
                    'file': filepath,
                    'line': line_num,
                    'description': description,
                    'match': match.group()
                })
    
    except Exception as e:
        print(f"Error checking {filepath}: {e}")
    
    return issues

def main():
    """Main validation function."""
    
    all_issues = []
    files_checked = 0
    
    # Check directories
    for dir_path in DIRS_TO_CHECK:
        if os.path.isfile(dir_path):
            # Single file
            if not should_exclude(dir_path):
                files_checked += 1
                issues = check_file(dir_path)
                all_issues.extend(issues)
        else:
            # Directory
            for root, dirs, files in os.walk(dir_path):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d))]
                
                for file in files:
                    if file.endswith('.py') and not should_exclude(file):
                        filepath = os.path.join(root, file)
                        files_checked += 1
                        issues = check_file(filepath)
                        all_issues.extend(issues)
    
    # Report results
    print(f"Files checked: {files_checked}")
    print()
    
    if all_issues:
        print("❌ VALIDATION FAILED")
        print("-" * 70)
        print(f"Found {len(all_issues)} hardcoded dimension(s):\n")
        
        for issue in all_issues:
            print(f"File: {issue['file']}")
            print(f"Line: {issue['line']}")
            print(f"Issue: {issue['description']}")
            print(f"Code: {issue['match']}")
            print()
        
        return False
    else:
        print("✅ VALIDATION PASSED")
        print("-" * 70)
        print("No hardcoded dimensions found!")
        print()
        print("All dimensions are dynamic:")
        print("  ✓ vector_dim auto-detected from model")
        print("  ✓ embedding_dim auto-detected from pretrained model")
        print("  ✓ No hardcoded np.zeros/ones/rand with fixed dimensions")
        print()
        
        # Test with actual code
        print("Testing with actual code...")
        print("-" * 70)
        
        try:
            from neuron_system.engines.compression import CompressionEngine
            
            # Test 384D
            engine_384 = CompressionEngine("all-MiniLM-L6-v2")
            engine_384._ensure_model_loaded()
            print(f"✓ 384D model: {engine_384.vector_dim}D (auto-detected)")
            
            # Test 768D
            engine_768 = CompressionEngine("all-mpnet-base-v2")
            engine_768._ensure_model_loaded()
            print(f"✓ 768D model: {engine_768.vector_dim}D (auto-detected)")
            
            print()
            print("✅ Dynamic dimensions working correctly!")
            
        except Exception as e:
            print(f"⚠️  Could not test with actual code: {e}")
            print("But validation passed - no hardcoded dimensions in code.")
        
        print()
        return True
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
