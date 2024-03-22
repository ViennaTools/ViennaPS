---
layout: default
title: Installing the Library
nav_order: 3
---

# Installing the Library
{: .fs-9 .fw-700}

---

## Supported Operating Systems

* Windows (Visual Studio)

* Linux (g++ / clang)

* macOS (XCode)

## System Requirements

* C++17 Compiler with OpenMP support

## Installing

The CMake configuration automatically checks if the dependencies are installed. If CMake is unable to find them, the dependencies will be built from source with the _buildDependencies_ target. Notably, ViennaPS operates as a header-only library, eliminating the need for a formal installation process. Nonetheless, we advise following the outlined procedure to neatly organize and relocate all header files to a designated directory:

```bash
git clone https://github.com/ViennaTools/ViennaPS.git
cd ViennaPS

cmake -B build && cmake --build build
cmake --install build --prefix "/path/to/your/custom/install/"
```

This will install the necessary headers and CMake files to the specified path. If `--prefix` is not specified, it will be installed to the standard path for your system, usually `/usr/local/` . 

{: .note}
> ViennaLS uses [VTK](https://gitlab.kitware.com/vtk/vtk) as dependency which can be installed beforehand to save some time when building the dependencies. On Linux based systems, it can be installed using the package manager: `sudo apt install libvtk9.1 libvtk9-dev`. On macOS, one can use Homebrew to install it: `brew install vtk`.

## Building the Python package

In order to build the Python bindings, the [pybind11](https://github.com/pybind/pybind11) library is required. On Linux based system (Ubuntu/Debian), pybind11 can be installed via the package manager: `sudo apt install pybind11-dev`. For macOS, the installation via Homebrew is recommended: `brew install pybind11`. 
The ViennaPS Python package can be built and installed using the `pip` command:

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

We recommend using [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) to consume this library.

* Installation with CPM
  ```cmake
  CPMAddPackage("gh:viennatools/viennaps@2.0.0")
  ```

* With a local installation
    > In case you have ViennaPS installed in a custom directory, make sure to properly specify the [`CMAKE_PREFIX_PATH`](https://cmake.org/cmake/help/latest/envvar/CMAKE_PREFIX_PATH.html#envvar:CMAKE_PREFIX_PATH).

    ```cmake
    list(APPEND CMAKE_PREFIX_PATH "/your/local/installation")

    find_package(ViennaPS)
    target_link_libraries(${PROJECT_NAME} PUBLIC ViennaTools::ViennaPS)
    ```