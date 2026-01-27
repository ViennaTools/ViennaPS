---
layout: default
title: GPU Module
parent: Installing the Library
nav_order: 1
---

# Installing the GPU Module 
{: .fs-9 .fw-500 }

---

## Requirements

The GPU ray tracing module is implemented using [**OptiX 8.0**](https://developer.nvidia.com/rtx/ray-tracing/optix). To use it, ensure your system meets the following requirements:

- **NVIDIA Driver:** Version 570 or higher
- **CUDA Toolkit:** Version 12+ (with compatible host compiler)

{: .note }
> ViennaPS depends on ViennaLS. When building ViennaPS locally (especially with GPU support), **you must also build ViennaLS locally** from the same source. Using the PyPI version of ViennaLS is **not compatible** with a local ViennaPS build.

---

## Python Bindings Installation

For a convenient setup, a helper script is provided. It builds **ViennaPS** and **ViennaLS** with GPU support directly from source inside the `ViennaTools` folder.

Run:

```sh
wget https://raw.githubusercontent.com/ViennaTools/ViennaPS/refs/tags/v4.2.1/python/scripts/install_ViennaTools.py && python3 install_ViennaTools.py
```

The script performs the following steps:

* Creates a virtual environment (`.venv`) in the `ViennaTools` directory.
* Builds and installs **ViennaLS** and **ViennaPS** with GPU support enabled.
* Installs required system dependencies (**VTK**, **Embree**).

> **Note:** Installing system dependencies requires `sudo` privileges.

---

There are two installation scripts available in the `python/scripts` directory, with different compatibility and functionality:

### 1. `install_ViennaPS.py` 

- **Compatibility:** All Linux distributions and Windows 
- **Functionality:**  
   - Builds and installs **ViennaPS** locally
   - Checks for an existing local build of **ViennaLS**
   - Checks for OptiX installation, downloads if not found
- **Limitations:**  
   - Assumes you have already installed dependencies like VTK and embree manually. Otherwise, they will be built from source, which can take a long time.

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

## CMake Configuration

To enable GPU support during CMake configuration, follow these steps:

1. Install the **CUDA toolkit system-wide** so CMake can detect it automatically or provide `CUDA_PATH` CMake variable.
2. Run CMake with the `-DVIENNAPS_USE_GPU=ON` option to enable GPU support. If the **CUDA Toolkit** is found, the GPU extension will be enabled.
   ```sh
   cmake -DVIENNAPS_USE_GPU=ON -B build
   ```
3. (Optional) To build examples or tests, set:
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
  VERSION 4.2.1
  GIT_REPOSITORY "https://github.com/ViennaTools/ViennaPS"
  OPTIONS "VIENNAPS_USE_GPU ON")

# Link against ViennaPS
add_executable(example_gpu main.cpp)
target_link_libraries(example_gpu PRIVATE ViennaPS)
```
