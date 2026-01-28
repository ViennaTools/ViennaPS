#!/usr/bin/env python3
"""Test script to verify ViennaPS installation in container."""

import sys


def test_viennaps_import():
    """Test that viennaps can be imported."""
    try:
        import viennaps as vps
        print(f"[PASS] ViennaPS imported successfully")
        print(f"       Module: {vps}")
        return True
    except ImportError as e:
        print(f"[FAIL] Could not import viennaps: {e}")
        return False


def test_viennaps_dimension():
    """Test that dimension can be set."""
    try:
        import viennaps as vps
        vps.setDimension(2)
        print("[PASS] ViennaPS dimension set to 2D")
        vps.setDimension(3)
        print("[PASS] ViennaPS dimension set to 3D")
        return True
    except Exception as e:
        print(f"[FAIL] Could not set dimension: {e}")
        return False


def test_viennaps_domain():
    """Test basic domain creation."""
    try:
        import viennaps as vps
        vps.setDimension(2)

        # Test creating a basic domain
        domain = vps.Domain()
        print(f"[PASS] Domain created: {domain}")
        return True
    except Exception as e:
        print(f"[FAIL] Could not create domain: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ViennaPS Container Installation Tests")
    print("=" * 60)

    results = []

    results.append(("Import", test_viennaps_import()))
    results.append(("Dimension", test_viennaps_dimension()))
    results.append(("Domain", test_viennaps_domain()))

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
