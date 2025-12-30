#!/usr/bin/env python3
"""Simple test runner for Flashlight tests."""
import sys
import subprocess

def run_tests():
    """Run pytest on the test suite."""
    # Add parent directory to path
    import os
    test_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(test_dir))
    sys.path.insert(0, parent_dir)
    
    # Run pytest
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_dir, "-v"],
        cwd=parent_dir
    )
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())

