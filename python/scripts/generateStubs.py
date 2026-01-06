import sys
import argparse

try:
    import pybind11_stubgen as stubgen
except ImportError:
    print(
        "pybind11-stubgen is not installed. Please install it using 'pip install pybind11-stubgen'."
    )
    sys.exit(1)


if __name__ == "__main__":
    # parse dim
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", type=str, default="python")
    args = parser.parse_args()

    # Don't create __pycache__ directory
    sys.dont_write_bytecode = True

    package_name = "viennaps"

    stubgen.main([package_name, "-o", args.dir, "--ignore-all-errors"])
