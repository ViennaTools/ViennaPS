# ViennaTools Installation Scripts

This directory contains scripts for installing ViennaTools (ViennaLS and ViennaPS) on various platforms.

## Scripts Available

### `install_ViennaTools.py` (Recommended)
Cross-platform installation script that supports both Linux distributions and macOS.

**Supported Platforms:**
- **Linux:** Ubuntu 22.04+, Debian 11+, Fedora 35+, Rocky Linux 8+, AlmaLinux 8+, Arch Linux, Manjaro, openSUSE Leap 15.3+, openSUSE Tumbleweed
- **macOS:** macOS 12+ (Monterey and later) with Homebrew

## Prerequisites

### For Linux:
- `sudo` privileges for installing system packages
- Git
- Python 3.8+

### For macOS:
- [Homebrew](https://brew.sh/) package manager
- Xcode Command Line Tools (will be installed automatically if missing)
- Git (usually comes with Xcode Command Line Tools)
- Python 3.8+

### For GPU Support (Optional):
- NVIDIA CUDA Toolkit 12.0+

## Usage

### Basic Installation (CPU-only)
```bash
./install_ViennaTools.py --no-gpu
```

### GPU-enabled Installation
```bash
./install_ViennaTools.py --gpu
```

### Installation with Specific Versions
```bash
./install_ViennaTools.py --viennals-version v1.2.0 --viennaps-version v2.1.0
```

### Verbose Installation
```bash
./install_ViennaTools.py -v
```

## Command Line Options

- `-v, --verbose`: Enable verbose output during installation
- `--gpu`: Enable GPU support (default: enabled)
- `--no-gpu`: Disable GPU support
- `--viennals-version VERSION`: Specify ViennaLS version (tag name or 'master')
- `--viennaps-version VERSION`: Specify ViennaPS version (tag name or 'master')

## macOS-Specific Notes

### Installing Homebrew
If you don't have Homebrew installed, run:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### GPU Support on macOS
GPU support on macOS is currently not available due to the lack of NVIDIA GPU support.

## Installation Process

The script performs the following steps:

1. **Platform Detection**: Automatically detects your operating system and version
2. **Dependency Installation**: Installs required system packages using the appropriate package manager
3. **Repository Cloning**: Downloads ViennaLS and ViennaPS source code
4. **Virtual Environment**: Creates a Python virtual environment
5. **Package Installation**: Builds and installs ViennaTools in the virtual environment

## Package Managers Used

- **Ubuntu/Debian**: `apt`
- **Fedora/RHEL/Rocky/Alma**: `dnf` or `yum`
- **Arch/Manjaro**: `pacman`
- **openSUSE**: `zypper`
- **macOS**: `brew` (Homebrew)

## Installed Dependencies

### Linux Dependencies:
- Build tools (gcc, g++, cmake, make)
- VTK development libraries
- Embree ray tracing library
- Python development tools

### macOS Dependencies:
- Xcode Command Line Tools (clang, clang++)
- CMake
- VTK
- Embree

## Virtual Environment

The script creates a Python virtual environment in the `ViennaTools/.venv` directory. To activate it after installation:

```bash
source ViennaTools/.venv/bin/activate
```

To deactivate:
```bash
deactivate
```

## Troubleshooting

### Common Issues:

1. **Missing Homebrew on macOS**: Install Homebrew first
2. **CUDA not found**: Ensure CUDA toolkit is properly installed and `nvcc` is in PATH
3. **Permission denied**: Ensure the script is executable (`chmod +x install_ViennaTools.py`)

### Getting Help:
- Check that your platform is supported
- Ensure all prerequisites are installed
- Run with `-v` flag for verbose output
- Check the GitHub issues for ViennaLS and ViennaPS repositories

## Examples

### Complete GPU Installation on Ubuntu:
```bash
# Install CUDA toolkit first (from NVIDIA)
./install_ViennaTools.py --gpu -v
```

### CPU-only Installation on macOS:
```bash
# Install Homebrew first
./install_ViennaTools.py --no-gpu
```

### Development Installation:
```bash
./install_ViennaTools.py --viennals-version master --viennaps-version master -v
```