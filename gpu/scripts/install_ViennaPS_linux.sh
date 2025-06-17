#!/bin/bash

# Before running this script, ensure you have the following dependencies installed:
# 1. Embree >= 4.0.0 
# 2. VTK >= 9.0.0
# (on Ubuntu 24.04, you can install them using the command: sudo apt install -y libvtk9-dev libembree-dev)

# Check if the script is run with bash
if [ -z "$BASH_VERSION" ]; then
    echo "This script must be run with bash. Please run it using: bash install_ViennaPS_linux.sh <-v|--verbose>"
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

# Directory for the virtual environment
read -r -p "Enter the path to the virtual environment directory (default: .venv): " venv_dir
if [ -z "$venv_dir" ]; then
    venv_dir=".venv"
    echo "No virtual environment directory specified. Using default: $venv_dir"
fi

# Check if gcc-12 and g++-12 are installed
if ! command -v gcc-12 &> /dev/null || ! command -v g++-12 &> /dev/null; then
    echo "gcc-12 and g++-12 are required but not installed. Please install them first."
    echo "You can install them using the following command:"
    echo "sudo apt install -y gcc-12 g++-12"
    exit 1
fi

# Check CUDA version
if ! command -v nvcc &> /dev/null; then
    echo "nvcc is required but not installed. Please install CUDA toolkit first."
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

# Check if venv directory exists
if [ -d "$venv_dir" ]; then
    echo "$venv_dir already exists. Reusing the existing virtual environment."
else
    # Create a new Python virtual environment
    python3 -m venv $venv_dir
fi
    
# Activate the virtual environment
source $venv_dir/bin/activate

# Check if ViennaLS Python package is installed
viennals_version_required="4.3.2"
if ! pip show ViennaLS &> /dev/null; then
    echo "ViennaLS Python package is not installed. Local ViennaLS build is required."
    read -r -p "Enter the path to the ViennaLS directory (e.g., /path/to/ViennaLS, press enter to download): " viennals_dir
    if [ -z "$viennals_dir" ]; then
        echo "No ViennaLS directory specified. Downloading version $viennals_version_required."
        mkdir ViennaLS_tmp_install
        git clone https://github.com/ViennaTools/ViennaLS.git ViennaLS_tmp_install
        cd ViennaLS_tmp_install || { echo "Failed to change directory to ViennaLS_tmp_install"; exit 1; }
        git checkout v$viennals_version_required &> /dev/null
        CC=gcc-12 CXX=g++-12 pip install . $verbose_flag
        cd ..
        rm -rf ViennaLS_tmp_install
    fi
    cd "$viennals_dir" || { echo "Failed to change directory to $viennals_dir"; exit 1; }
    CC=gcc-12 CXX=g++-12 pip install . $verbose_flag
else
    viennals_version=$(pip show ViennaLS | grep Version | awk '{print $2}')
    echo -e "\033[1;33m""ViennaLS Python package is already installed ($viennals_version). Local build is required.\033[0m"
    echo -e "\033[1;33m""If you want to reinstall, please uninstall it first using: pip uninstall ViennaLS\033[0m"
    read -r -p "Continue with the current version ($viennals_version)? (y/n): " continue_install
    if [[ "$continue_install" != "y" && "$continue_install" != "Y" ]]; then
        echo "Installation aborted."
        exit 1
    fi
    if [ "$viennals_version" != "$viennals_version_required" ]; then
        echo -e "\033[1;33m""Note: The version of ViennaLS is not $viennals_version_required, which may cause compatibility issues.\033[0m"
        echo -e "\033[1;33m""Please ensure you have the correct version installed.\033[0m"
        exit 1
    fi
fi

# Check if current directory is ViennaPS
viennaps_dir=$(basename "$PWD")
if [ "$viennaps_dir" != "ViennaPS" ]; then 
    read -r -p "Please enter the path to the ViennaPS directory (e.g., /path/to/ViennaPS): " viennaps_dir
    if [ -z "$viennaps_dir" ]; then
        echo "No ViennaPS directory specified. Please provide a path."
        exit 1
    fi
    cd "$viennaps_dir" || { echo "Failed to change directory to $viennaps_dir"; exit 1; }
fi

# Install ViennaPS with GPU support (using gcc-12 and g++-12)
OptiX_INSTALL_DIR=$optix_dir CC=gcc-12 CXX=g++-12 CMAKE_ARGS=-DVIENNAPS_FORCE_GPU=ON pip install . $verbose_flag

echo "Installation complete. To activate the virtual environment, run:"
echo "source $venv_dir/bin/activate"
echo "To deactivate the virtual environment, run:"
echo "deactivate"