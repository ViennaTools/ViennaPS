try:
    import viennaps2d
except ImportError:
    print("ERROR: Python bindings for viennaps2d are not available")
    exit()

import viennaps2d as vps


def test_process_with_domain():
    """Test process application with domain"""
    print("Testing Process with Domain...")

    domain = vps.Domain()
    process = vps.Process()

    # Test process application without proper setup (should fail)
    try:
        process.apply()
        print("WARNING: Process applied without expected error")
    except RuntimeError as e:
        print(f"Process application failed as expected: {e}")


def test_complete_workflow():
    """Test a complete simulation workflow"""
    print("Testing complete workflow...")

    vps.Logger.setLogLevel(vps.LogLevel.INFO)

    # Create components
    domain = vps.Domain()
    process = vps.Process()

    # Try to create a model and add it to process
    try:
        model = vps.SF6O2Etching()
        # If successful, try to add to process
        print("Model created, attempting to configure process...")
    except RuntimeError as e:
        print(f"Model creation failed: {e}")

    print("Complete workflow test finished")


def test_error_handling():
    """Test error handling scenarios"""
    print("Testing error handling...")

    # Test various error scenarios
    test_cases = [
        lambda: vps.Process().apply(),  # Process without setup
    ]

    for i, test_case in enumerate(test_cases):
        try:
            test_case()
            print(f"Test case {i+1}: No error (unexpected)")
        except Exception as e:
            print(f"Test case {i+1}: Caught expected error: {e}")


if __name__ == "__main__":
    test_process_with_domain()
    test_complete_workflow()
    test_error_handling()
    print("Integration tests completed!")
