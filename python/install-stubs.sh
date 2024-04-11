#!/bin/bash

# experimental script to install stubs for python packages
# this script is not yet integrated into the main build process

# install stubs for all packages in the current environment
# this is useful for IDEs that support type hinting

# TODO: This should be done by the GitHub Workflow
# TODO: that produces the published wheels

path=$(python3 -m site --user-site)
echo "Installing stubs in $path"
cp -r stubs/* $path
