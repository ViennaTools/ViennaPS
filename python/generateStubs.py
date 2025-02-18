import mypy.stubgen as stubgen
import sys
import argparse


if __name__ == "__main__":
    # parse dim
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", type=int, default=2)
    args = parser.parse_args()
    dim = args.D

    # Don't create __pycache__ directory
    sys.dont_write_bytecode = True

    # Initialize the stubgen parser options
    options = stubgen.parse_options(
        [
            "-o",
            "stubs_ls",
            "-p",
            "viennals" + str(dim) + "d",
        ]
    )

    # Generate the stubs
    stubgen.generate_stubs(options)
