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

    package_name = "viennaps" + str(args.D) + "d"

    stubgen.main([package_name, "-o", args.dir, "--ignore-all-errors"])
