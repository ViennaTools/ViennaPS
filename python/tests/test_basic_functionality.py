try:
    import viennaps2d
except ImportError:
    print("ERROR: Python bindings for viennaps2d are not available")
    exit()

import viennaps2d as vps


def test_logger():
    """Test logger functionality"""
    print("Testing Logger...")
    vps.Logger.setLogLevel(vps.LogLevel.INFO)
    vps.Logger.setLogLevel(vps.LogLevel.DEBUG)
    vps.Logger.setLogLevel(vps.LogLevel.WARNING)
    vps.Logger.setLogLevel(vps.LogLevel.ERROR)
    print("Logger test passed")


def test_domain_creation():
    """Test domain creation and basic operations"""
    print("Testing Domain creation...")
    domain = vps.Domain()
    assert domain is not None
    print("Domain creation test passed")


def test_process_creation():
    """Test process creation"""
    print("Testing Process creation...")
    process = vps.Process()
    assert process is not None
    print("Process creation test passed")


if __name__ == "__main__":
    test_logger()
    test_domain_creation()
    test_process_creation()
    print("All basic functionality tests passed!")
