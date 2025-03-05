---
layout: default
title: GPU Acceleration
parent: Installing the Library
nav_order: 1
---

# Installing the GPU Module for ViennaPS 3D
{: .fs-9 .fw-500 }

---


## Requirements

The GPU ray tracing module is implemented using **OptiX 8.0**. To use it, ensure your system meets the following requirements:

- **NVIDIA Driver:** Version 535 or higher
- **CUDA Toolkit:** Version 12.0

## CMake Configuration

To enable GPU support, follow these steps:

1. Open the `CMakeLists.txt` file and enable the `VIENNAPS_USE_GPU` option.
2. Specify the path to your OptiX installation by setting the CMake variable `OptiX_INSTALL_DIR`.
   - This should point to the directory where the `include` folder containing OptiX headers is located.
3. Install the **CUDA toolkit system-wide** so CMake can detect it automatically or provide `CUDA_PATH` CMake variable.
4. Run CMake configuration. If both **CUDA and OptiX** are found, the GPU extension will be enabled.
5. (Optional) To build examples or tests, set:
   - `VIENNAPS_BUILD_EXAMPLES=ON`
   - `VIENNAPS_BUILD_TESTS=ON`

## Python Build Instructions

For building the Python GPU module:

1. Set the environment variable `OptiX_INSTALL_DIR` to the OptiX installation directory.
   - This variable will be used during the CMake setup.
2. Run the following command to install the module locally:
   ```sh
   pip install .
   ```
   Ensure `VIENNAPS_USE_GPU` is enabled and `OptiX_INSTALL_DIR` is set.
3. The GPU functions are available in the `GPU` submodule:
   ```python
   import viennaps3d as vps
   gpuProcess = vps.gpu.Process()
   ```
   **Note:** The GPU submodule is only available in the **3D bindings** since GPU ray tracing is implemented for **3D only**.


