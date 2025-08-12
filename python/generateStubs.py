import pybind11_stubgen as stubgen
import sys
import argparse


if __name__ == "__main__":
    # parse dim
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", type=int, default=2)
    parser.add_argument("-dir", type=str, default=".")
    args = parser.parse_args()

    # Don't create __pycache__ directory
    sys.dont_write_bytecode = True

    if args.D == 2:
        stubgen.main(["viennaps2d", "-o", args.dir, "--ignore-all-errors"])
    elif args.D == 3:
        stubgen.main(["viennaps3d", "-o", args.dir, "--ignore-all-errors"])
    else:
        raise ValueError("Dimension D must be either 2 or 3.")
