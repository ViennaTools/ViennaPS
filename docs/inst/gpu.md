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

The Python package can be built with GPU support using helper scripts in
`python/scripts`. GPU support enables GPU ray tracing in ViennaPS and the GPU
BiCGSTAB solver used by the oxidation model.

### Existing ViennaPS Checkout

Use `install_ViennaPS.py` when working from a ViennaPS checkout. It creates or
reuses a virtual environment, installs a compatible local ViennaLS build, and
then installs ViennaPS from the selected checkout.

```sh
python python/scripts/install_ViennaPS.py
```

When working from existing local ViennaPS and ViennaLS checkouts, pass the
ViennaLS source directory explicitly:

```sh
python python/scripts/install_ViennaPS.py --viennals-dir ../ViennaLS
```

Use `--no-gpu` for a CPU-only local build.

### Fresh ViennaTools Setup

Use `install_ViennaTools.py` for a broader fresh setup. It creates a
`ViennaTools` directory, installs supported system dependencies, clones
ViennaLS and ViennaPS, creates a virtual environment, and builds both Python
packages.

From a ViennaPS checkout:

```sh
python python/scripts/install_ViennaTools.py
```

Or download the script directly from a tagged release:

```sh
wget https://raw.githubusercontent.com/ViennaTools/ViennaPS/refs/tags/v4.6.1/python/scripts/install_ViennaTools.py
python3 install_ViennaTools.py
```

GPU support is enabled by default on Linux when CUDA is available. Use
`--no-gpu` for a CPU-only setup. On macOS, GPU support is disabled because
NVIDIA CUDA/OptiX is not available.

{: .note }
Installing system dependencies with `install_ViennaTools.py` may require
administrator privileges on Linux.

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
  VERSION 4.6.1
  GIT_REPOSITORY "https://github.com/ViennaTools/ViennaPS"
  OPTIONS "VIENNAPS_USE_GPU ON")

# Link against ViennaPS
add_executable(example_gpu main.cpp)
target_link_libraries(example_gpu PRIVATE ViennaTools::ViennaPS)
```
