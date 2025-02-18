#!/bin/bash

# experimental script to install stubs for python packages
# this script is not yet integrated into the main build process

# install stubs for all packages in the current environment
# this is useful for IDEs that support type hinting

# which python3
path=$(python3 -c 'import site; print(site.getsitepackages()[0])')
echo "Installing stubs in $path"
cp -r viennaps2d/* "$path/viennaps2d/"
cp -r viennaps3d/* "$path/viennaps3d/"