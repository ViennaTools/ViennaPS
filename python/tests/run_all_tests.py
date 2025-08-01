import sys
import os
import importlib.util

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_test_file(filename):
    """Run a test file and catch any errors"""
    print(f"\n{'='*50}")
    print(f"Running {filename}")
    print("=" * 50)

    try:
        # Load and execute the module properly
        spec = importlib.util.spec_from_file_location("test_module", filename)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        print(f"✓ {filename} completed successfully")
        return True
    except Exception as e:
        print(f"✗ {filename} failed with error: {e}")
        return False


def main():
    """Run all test files"""
    test_files = [
        "test_basic_functionality.py",
        "test_models.py",
        "test_integration.py",
    ]

    print("ViennaPS Python Bindings Test Suite")
    print("=" * 50)

    results = {}
    for test_file in test_files:
        if os.path.exists(test_file):
            results[test_file] = run_test_file(test_file)
        else:
            print(f"Warning: {test_file} not found, skipping...")
            results[test_file] = False

    # Summary
    print(f"\n{'='*50}")
    print("Test Summary")
    print("=" * 50)

    passed = sum(results.values())
    total = len([f for f in test_files if os.path.exists(f)])

    for test_file, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{test_file:<30} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
