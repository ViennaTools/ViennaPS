#!/bin/bash

# experimental script to install stubs for python packages
# this script is not yet integrated into the main build process

# install stubs for all packages in the current environment
# this is useful for IDEs that support type hinting

path=$(python3 -m site --user-site)
echo "Installing stubs in $path"
cp -r stubs/* $path