#!/bin/bash

# This script installs the ViennaTools package on Ubuntu 24.04 with GPU support.
# It attempts to install the required dependencies on the system, therefore sudo privileges are required.

# Check if the script is run with bash
if [ -z "$BASH_VERSION" ]; then
    echo "This script must be run with bash. Please run it using: bash install_ViennaTools_ubuntu.sh <-v|--verbose>"
    exit 1
fi

# Check if verbose mode is enabled
verbose_flag=""
if [[ "$1" == "-v" || "$1" == "--verbose" ]]; then
    echo "Verbose mode is enabled."
    verbose_flag="-v"
else
    echo "Verbose mode is disabled."
fi

# Check ubuntu version
if [ "$(lsb_release -rs)" != "24.04" ]; then
    echo "This script is intended for Ubuntu 24.04. Please run it on the correct version."
    exit 1
fi
echo "Ubuntu version is 24.04. Proceeding with installation."

read -r -p "Enter the path to the virtual environment directory (default: .venv): " venv_dir
if [ -z "$venv_dir" ]; then
    venv_dir=".venv"
    echo "No virtual environment directory specified. Using default: $venv_dir"
fi

# Check if gcc-12 and g++-12 are installed
if ! command -v gcc-12 &> /dev/null || ! command -v g++-12 &> /dev/null; then
    echo "gcc-12 and g++-12 are required but not installed. Please install them first."
    exit 1
fi

# Check CUDA version
if ! command -v nvcc &> /dev/null; then
    echo "nvcc is required but not installed. Please install it first."
    exit 1
fi
nvcc_version=$(nvcc --version | grep release | sed 's/.*release //; s/,//')
if [[ $nvcc_version < "12.0" ]]; then
    echo "CUDA toolkit version 12.0 or higher is required. Please update your CUDA toolkit."
    exit 1
fi
echo "CUDA toolkit version: $nvcc_version"

# Check for OptiX directory
if [ -n "$OptiX_INSTALL_DIR" ]; then
    optix_dir="$OptiX_INSTALL_DIR"
    echo "Using OptiX directory from environment variable: $optix_dir"
fi
if [ -z "$optix_dir" ]; then
    read -r -p "Please enter the path to the OptiX directory (e.g., /path/to/Optix): " optix_dir
    if [ -z "$optix_dir" ]; then
        echo "No OptiX directory specified. Please set the OptiX_INSTALL_DIR environment variable or provide a path."
        exit 1
    fi
    echo "OptiX directory is set to: $optix_dir"
fi

# Install VTK and embree and python3-venv
sudo apt install -y libvtk9-dev libembree-dev python3-venv

# Check if the ViennaTools directory already exists
if [ -d "ViennaTools" ]; then
    echo "ViennaTools directory exists. Attempting to reinstall."

    cd ViennaTools
    cd ViennaLS
    if [ $? -ne 0 ]; then
        echo "Failed to navigate to ViennaLS directory."
        exit 1
    fi
    git pull
    cd ..

    cd ViennaPS
    if [ $? -ne 0 ]; then
        echo "Failed to navigate to ViennaPS directory."
        exit 1
    fi
    git pull
    cd ..

    # Check if venv directory exists
    if [ -d "$venv_dir" ]; then
        echo "$venv_dir already exists. Reusing the existing virtual environment."
    else
        # Create a new Python virtual environment
        python3 -m venv $venv_dir
    fi
    
else
    # Create a directory for the installation
    mkdir -p ViennaTools
    cd ViennaTools
    if [ $? -ne 0 ]; then
        echo "Failed to create or navigate to ViennaTools directory."
        exit 1
    fi

    # Clone the ViennaLS repository
    git clone https://github.com/ViennaTools/ViennaLS.git
    if [ $? -ne 0 ]; then
        echo "Failed to clone ViennaLS repository."
        exit 1
    fi

    # Clone the ViennaPS repository
    git clone https://github.com/ViennaTools/ViennaPS.git
    if [ $? -ne 0 ]; then
        echo "Failed to clone ViennaPS repository."
        exit 1
    fi

    # Create Python virtual environment
    python3 -m venv $venv_dir
fi

# Activate the virtual environment
source $venv_dir/bin/activate

# Install ViennaLS
cd ViennaLS
CC=gcc-12 CXX=g++-12 pip install . $verbose_flag
cd ..

# Install ViennaPS with GPU support (using gcc-12 and g++-12)
cd ViennaPS
OptiX_INSTALL_DIR=$optix_dir CC=gcc-12 CXX=g++-12 CMAKE_ARGS=-DVIENNAPS_FORCE_GPU=ON pip install . $verbose_flag
cd ..

echo "Installation complete. To activate the virtual environment, run:"
echo "source ViennaTools/$venv_dir/bin/activate"
echo "To deactivate the virtual environment, run:"
echo "deactivate"