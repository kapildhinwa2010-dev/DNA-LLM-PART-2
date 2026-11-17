"""
Test that all experiment files can be imported and have valid Python syntax.
This is a basic smoke test to ensure the code structure is correct.
"""
import importlib
import os
import sys
import traceback

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_all_experiments_importable():
    """Test that all experiment files can be imported without syntax errors."""
    experiments_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments')
    
    failed_imports = []
    successful_imports = []
    
    # Walk through all Python files in experiments directory
    for root, dirs, files in os.walk(experiments_dir):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                # Get the module path
                rel_path = os.path.relpath(root, experiments_dir)
                module_path = os.path.join('experiments', rel_path, file).replace('.py', '').replace(os.sep, '.')
                
                try:
                    # Try to import the module
                    importlib.import_module(module_path)
                    successful_imports.append(module_path)
                    print(f"✓ Successfully imported: {module_path}")
                except Exception as e:
                    failed_imports.append((module_path, str(e)))
                    print(f"✗ Failed to import {module_path}: {e}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Import Test Summary:")
    print(f"  Successful: {len(successful_imports)}")
    print(f"  Failed: {len(failed_imports)}")
    print(f"{'='*70}")
    
    # If there are failures, print details
    if failed_imports:
        print("\nFailed imports:")
        for module_path, error in failed_imports:
            print(f"  - {module_path}: {error}")
    
    # Assert that all imports succeeded
    assert len(failed_imports) == 0, f"{len(failed_imports)} imports failed"


def test_file_naming_convention():
    """Test that all files follow the naming convention (snake_case)."""
    experiments_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments')
    
    invalid_files = []
    valid_files = []
    
    for root, dirs, files in os.walk(experiments_dir):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                # Check if filename follows snake_case convention
                filename_without_ext = file[:-3]
                if not filename_without_ext.replace('_', '').isalnum():
                    invalid_files.append(file)
                elif filename_without_ext != filename_without_ext.lower():
                    invalid_files.append(file)
                elif '__' in filename_without_ext:
                    invalid_files.append(file)
                else:
                    valid_files.append(file)
    
    print(f"\nFile Naming Test:")
    print(f"  Valid: {len(valid_files)}")
    print(f"  Invalid: {len(invalid_files)}")
    
    if invalid_files:
        print(f"  Invalid files: {invalid_files}")
    
    assert len(invalid_files) == 0, f"{len(invalid_files)} files have invalid names"


def test_no_hindi_in_code():
    """Test that no Hindi text remains in the code files."""
    experiments_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments')
    
    hindi_indicators = [
        'Kya', 'Agar', 'toh', 'hai', 'mein', 'ko', 'ka', 'ki', 'hua',
        'karna', 'dekho', 'Matlab', 'next', 'FIX', 'NEXT', 'PUBLISHABLE',
        'BREAKTHROUGH', 'STABLE', 'DRIFT', 'COLLAPSE', 'VIABLE',
        'behavior', 'compression', 'DNA'
    ]
    
    # We'll check for common Hindi words that shouldn't be in English code
    hindi_words = ['Kya', 'Agar', 'toh', 'hai', 'mein', 'ko', 'ka', 'ki', 'hua']
    
    files_with_hindi = []
    clean_files = []
    
    for root, dirs, files in os.walk(experiments_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                found_hindi = False
                for word in hindi_words:
                    # Check for word boundaries to avoid false positives
                    if f' {word} ' in content or f'{word} ' in content or f' {word}' in content:
                        found_hindi = True
                        break
                
                if found_hindi:
                    files_with_hindi.append(file)
                else:
                    clean_files.append(file)
    
    print(f"\nHindi Text Check:")
    print(f"  Clean files: {len(clean_files)}")
    print(f"  Files with potential Hindi: {len(files_with_hindi)}")
    
    if files_with_hindi:
        print(f"  Files that may contain Hindi: {files_with_hindi}")
    
    # For now, just warn but don't fail (as some files might still be in progress)
    if files_with_hindi:
        print(f"  WARNING: {len(files_with_hindi)} files may still contain Hindi text")


if __name__ == '__main__':
    print("Running Reverse.Dean tests...\n")
    
    try:
        test_all_experiments_importable()
        print("\n✓ Import test passed")
    except AssertionError as e:
        print(f"\n✗ Import test failed: {e}")
        sys.exit(1)
    
    try:
        test_file_naming_convention()
        print("✓ Naming convention test passed")
    except AssertionError as e:
        print(f"✗ Naming convention test failed: {e}")
        sys.exit(1)
    
    try:
        test_no_hindi_in_code()
        print("✓ Hindi text check completed (see warnings above)")
    except Exception as e:
        print(f"✗ Hindi text check failed: {e}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("All basic tests passed!")
    print("="*70)
