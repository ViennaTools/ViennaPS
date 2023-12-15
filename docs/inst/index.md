---
layout: default
title: Installation
nav_order: 3
---

# Installation
{: .fs-9 .fw-700}

---

The CMake configuration automatically checks if the dependencies are installed. If CMake is unable to find them, the dependencies will be built from source with the _buildDependencies_ target. Notably, ViennaPS operates as a header-only library, eliminating the need for a formal installation process. Nonetheless, we advise following the outlined procedure to neatly organize and relocate all header files to a designated directory:

```bash
git clone https://github.com/ViennaTools/ViennaPS.git
cd ViennaPS
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/your/custom/install/
make buildDependencies # this will install all dependencies and might take a while
make install
```

This will install the necessary headers and CMake files to the specified path. If `CMAKE_INSTALL_PREFIX` is not specified, it will be installed to the standard path for your system, usually `/usr/local/` . 

If one wants to use a specific installation of one or more of the dependencies, just pass the corresponding _*_DIR_ variable as a configuration option (e.g. -DViennaLS_DIR=/path/to/viennals/install -DViennaRay_DIR=/path/to/viennaray/install).

## Building the Python package

The Python package can be built and installed using the `pip` command:

```bash
git clone https://github.com/ViennaTools/ViennaPS.git
cd ViennaPS
pip install --user .
```

{: .note}
> Some functionalities of the ViennaPS Python module only work in combination with the ViennaLS Python module. It is therefore recommended to additionally install the ViennaLS Python module on your system. Instructions to do so can be found in the [ViennaLS Git Repository](https://github.com/ViennaTools/viennals).

## Using the Python package

The 2D version of the library can be imported as follows:
```python
import viennaps2d as vps
```

In order to switch to three dimensions, only the import needs to be changed:

```python
import viennaps3d as vps
```

## Integration in CMake projects

In order to use this library in your CMake project, add the following lines to the CMakeLists.txt of your project:

```CMake
set(ViennaPS_DIR "/path/to/your/custom/install/")
find_package(ViennaPS REQUIRED)
add_executable(${PROJECT_NAME} ...)
target_include_directories(${PROJECT_NAME} PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${VIENNAPS_LIBRARIES})
``` 