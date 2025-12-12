---
layout: default
title: GPU Module
parent: Installing the Library
nav_order: 1
---

# Installing the GPU Module 
{: .fs-9 .fw-500 }

---

{: .warning }
> The GPU ray tracing module is still an **experimental** feature and is under active development. If you encounter any issues or have suggestions, please let us know on [GitHub](https://github.com/ViennaTools/ViennaPS/issues).

## Requirements

The GPU ray tracing module is implemented using [**OptiX 8.0**](https://developer.nvidia.com/rtx/ray-tracing/optix). To use it, ensure your system meets the following requirements:

- **NVIDIA Driver:** Version 570 or higher
- **CUDA Toolkit:** Version 12.0
- **GCC:** Version 12.0 

{: .note }
> ViennaPS depends on ViennaLS. When building ViennaPS locally (especially with GPU support), **you must also build ViennaLS locally** from the same source. Using the PyPI version of ViennaLS is **not compatible** with a local ViennaPS build.

---

## Python Bindings Installation

For a convenient setup, a helper script is provided. It builds **ViennaPS** and **ViennaLS** with GPU support directly from source inside the `ViennaTools` folder.

Run:

```sh
wget https://raw.githubusercontent.com/ViennaTools/ViennaPS/refs/tags/v4.1.2/gpu/scripts/install_ViennaTools.py && python3 install_ViennaTools.py
```

The script performs the following steps:

* Creates a virtual environment (`.venv`) in the `ViennaTools` directory.
* Builds and installs **ViennaLS** and **ViennaPS** with GPU support enabled.
* Installs required system dependencies (**VTK**, **Embree**).

> **Note:** Installing system dependencies requires `sudo` privileges.

There are two installation scripts available in the `gpu/scripts` directory, each with different compatibility and functionality:

### 1. `install_ViennaPS_linux.py` 

- **Compatibility:** All Linux distributions  
- **Functionality:**  
   - Builds and installs **ViennaPS** locally
   - Checks for an existing local build of **ViennaLS**
   - Checks for OptiX installation, downloads if not found
- **Limitations:**  
   - Assumes you have already installed dependencies like VTK and embree manually

### 2. `install_ViennaTools.py`

- **Compatibility:** 
  - Linux: Ubuntu 22.04+, Debian 11+, Fedora 35+, Rocky Linux 8+, AlmaLinux 8+, Arch Linux, Manjaro, openSUSE Leap 15.3+, openSUSE Tumbleweed
  - macOS: macOS 12+ (Monterey and later) with Homebrew (only CPU support, no GPU)
- **Prerequisites**:
  - For Linux:
    - `sudo` privileges for installing system packages
    - Git
    - Python 3.8+

  - For macOS:
    - [Homebrew](https://brew.sh/) package manager
    - Xcode Command Line Tools (will be installed automatically if missing)
    - Git (usually comes with Xcode Command Line Tools)
    - Python 3.8+
- **Functionality:**  
  - Installs all required dependencies: `VTK`, `embree`, and others using `apt`  
  - Builds and installs **ViennaLS** and **ViennaPS** in a local folder named `ViennaTools`  
  - Suitable for a fresh installation on Ubuntu systems  
- **Command Line Options**:
  - `-v, --verbose`: Enable verbose output during installation
  - `--gpu`: Enable GPU support (default: enabled)
  - `--no-gpu`: Disable GPU support
  - `--viennals-version VERSION`: Specify ViennaLS version (tag name or 'master')
  - `--viennaps-version VERSION`: Specify ViennaPS version (tag name or 'master')

## CMake Configuration

To enable GPU support, follow these steps:

1. Run CMake with the `-DVIENNAPS_USE_GPU=ON` option to enable GPU support.
2. Specify the path to your OptiX installation by setting the CMake variable `OptiX_INSTALL_DIR`.
   - This should point to the directory where the `include` folder containing OptiX headers is located.
  E.g.:
   ```sh
   cmake -DVIENNAPS_USE_GPU=ON -DOptiX_INSTALL_DIR=/path/to/optix .
   ```
   Alternatively, you can set the `OptiX_INSTALL_DIR` environment variable:
   ```sh
   export OptiX_INSTALL_DIR=/path/to/optix
   ```
   This will be used during the CMake setup.
3. Install the **CUDA toolkit system-wide** so CMake can detect it automatically or provide `CUDA_PATH` CMake variable.
4. Run CMake configuration. If both **CUDA and OptiX** are found, the GPU extension will be enabled.
5. (Optional) To build examples or tests, set:
   - `VIENNAPS_BUILD_EXAMPLES=ON`
   - `VIENNAPS_BUILD_TESTS=ON`

### CMake Example Project

Here is an example CMake project that demonstrates how to link against the ViennaPS GPU module using CPM to download ViennaPS:

```cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project("ExampleProject")

include("cmake/cpm.cmake") # Include CPM.cmake (get from: https://github.com/cpm-cmake/CPM.cmake/releases)

CPMFindPackage(
  NAME ViennaPS
  VERSION 4.1.2
  GIT_REPOSITORY "https://github.com/ViennaTools/ViennaPS"
  OPTIONS "VIENNAPS_USE_GPU ON")

add_gpu_executable(example target_name example.cpp)
```

## Custom Python Build Instructions

For building the Python GPU module:

1. Set the environment variable `OptiX_INSTALL_DIR` to the OptiX installation directory.
   - This variable will be used during the CMake setup.
2. Run the following command to install the module locally:
   ```sh
   CMAKE_ARGS=-DVIENNAPS_USE_GPU=ON pip install .
   ```
3. The GPU functions are available in the `GPU` submodule:
   ```python
   import viennaps3d as vps
   context = vps.gpu.Context()
   context.create()
   gpuProcess = vps.gpu.Process(context)
   ```
   **Note:** The GPU submodule is only available in the **3D bindings** since GPU ray tracing is implemented for **3D only**.

Example usage of the GPU module can be found in the [examples](https://github.com/ViennaTools/ViennaPS/tree/master/gpu/examples).

{: .note }
> Currently, only a limited number of models are available for GPU acceleration: [SingleParticleProcess]({% link models/prebuilt/singleParticle.md %}), [MultiParticleProcess]({% link models/prebuilt/multiParticle.md %}), and [SF6O2Etching]({% link models/prebuilt/SF6O2Etching.md %}).