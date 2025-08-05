---
layout: default
title: Installing the Library
nav_order: 3
has_children: true
---

# Installing the Library
{: .fs-9 .fw-700}

---

## Quick Start  

To install ViennaPS for Python, simply run:  

```sh
pip install ViennaPS
```
If there is no pre-built package available for your operating system, you can build the package yourself using the instructions below.
To use ViennaPS in C++, clone the repository and follow the installation steps below.

## Supported Operating Systems

* Windows (Visual Studio)

* Linux (g++ / clang)

* macOS (XCode)

## System Requirements

* C++17 Compiler with OpenMP support

## Installing

{: .note}
> ViennaLS uses [VTK](https://gitlab.kitware.com/vtk/vtk) as dependency which can be installed beforehand to save some time when building the dependencies. On Linux based systems, it can be installed using the package manager: `sudo apt install libvtk9.1 libvtk9-dev`. On macOS, one can use Homebrew to install it: `brew install vtk`.

{: .note}
> ViennaRay uses [Embree](https://github.com/RenderKit/embree) as dependency which can be installed beforehand to save some time when building the dependencies. On Linux based systems, it can be installed using the package manager: `sudo apt install libembree-dev`. If you are using an Ubuntu version older than 24, the installed package will be Embree version 3, and you must additionally pass `VIENNARAY_EMBREE_VERSION=3` to the CMake options, e.g., `cmake -B build -G Ninja -D VIENNARAY_EMBREE_VERSION=3`
On macOS, you can install Embree using Homebrew with the command: `brew install embree`.

The CMake configuration automatically checks if the dependencies are installed. If CMake is unable to find them, the dependencies will be built from source with the _buildDependencies_ target. Notably, ViennaPS operates as a header-only library, eliminating the need for a formal installation process. Nonetheless, we advise following the outlined procedure to neatly organize and relocate all header files to a designated directory:

```bash
git clone https://github.com/ViennaTools/ViennaPS.git
cd ViennaPS

cmake -B build -G Ninja && cmake --build build
cmake --install build --prefix "/path/to/your/custom/install/"
```

This will install the necessary headers and CMake files to the specified path. If `--prefix` is not specified, it will be installed to the standard path for your system, usually `/usr/local/` . 

The `-G Ninja` option can be omitted if you prefer to use _Unix Makefiles_ as the build system. However, this can potentially lead to conflicts when later installing the Python package using the pip installer, as pip always employs _Ninja_ as the build system.

## Building the Python package locally


In order to build the Python bindings, the [pybind11](https://github.com/pybind/pybind11) library is required. On Linux based system (Ubuntu/Debian), pybind11 can be installed via the package manager: `sudo apt install pybind11-dev`. For macOS, the installation via Homebrew is recommended: `brew install pybind11`. 
The ViennaPS Python package can be built and installed using the `pip` command:

```bash
git clone https://github.com/ViennaTools/ViennaPS.git
cd ViennaPS

pip install .
```

{: .note}
> Some functionalities of the ViennaPS Python module only work in combination with the ViennaLS Python module. It is therefore necessary to additionally install the ViennaLS Python module on your system. Instructions to do so can be found in the [ViennaLS Git Repository](https://github.com/ViennaTools/viennals).

## Integration in CMake projects

We recommend using [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) to consume this library.

* Installation with CPM
  ```cmake
  CPMAddPackage("gh:viennatools/viennaps@3.7.0")
  ```

* With a local installation
    > In case you have ViennaPS installed in a custom directory, make sure to properly specify the [`CMAKE_PREFIX_PATH`](https://cmake.org/cmake/help/latest/envvar/CMAKE_PREFIX_PATH.html#envvar:CMAKE_PREFIX_PATH).

    ```cmake
    list(APPEND CMAKE_PREFIX_PATH "/your/local/installation")

    find_package(ViennaPS)
    target_link_libraries(${PROJECT_NAME} PUBLIC ViennaTools::ViennaPS)
    ```

### Shared Library
In order to save build time during development, dynamically linked shared libraries can be used if ViennaPS was built with them. This is done by precompiling the most common template specialisations. In order to use shared libraries, use

```bash
cmake -B build -DVIENNALS_PRECOMPILE_HEADERS=ON
```

If ViennaPS was built with shared libraries and you use ViennaPS in your project (see above), CMake will automatically link them to your project.