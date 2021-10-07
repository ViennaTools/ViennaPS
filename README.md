# ViennaPS

ViennaPS is a header-only C++ process simulation library, which includes surface and volume representations, a ray tracer, and physical models for the simulation of microelectronic fabrication processes.

IMPORTANT NOTE: ViennaPS is under heavy development and improved daily. If you do have suggestions or find bugs, please let us know!

## Support

<!-- [Documentation](https://viennatools.github.io/ViennaPS/doxygen/html/index.html) and [Examples](https://viennatools.github.io/ViennaPS/doxygen/html/examples.html) can be found online. -->

Bug reports and suggestions should be filed on GitHub.

<!-- ## Releases
Releases are tagged on the maser branch and available in the [releases section](https://github.com/ViennaTools/ViennaPS/releases). -->

## Building

### Supported Operating Systems

* Windows (Visual Studio)

* Linux (g++ / clang)

* macOS (XCode)


### System Requirements

* C++ Compiler with OpenMP support

* [ViennaHRLE](https://github.com/ViennaTools/viennahrle)

* [ViennaLS](https://github.com/ViennaTools/viennals)

* [ViennaRay](https://github.com/ViennaTools/viennaray)

<!-- ## Using ViennaPS in your project

Have a look at the [example repo](https://github.com/ViennaTools/viennals-example) for creating a project with ViennaPS as a dependency. -->


<!-- ## Installing (with dependencies already installed)

Since this is a header only project, it does not require any installation.
However, we recommend the following procedure.

Make sure you have [ViennaHRLE](https://github.com/ViennaTools/viennahrle) and [ViennaLS](https://github.com/ViennaTools/viennals) installed on your system and run:

```
git clone github.com/ViennaTools/ViennaPS.git
cd ViennaPS
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/your/custom/install/
make install
```

This will install the necessary headers and CMake files to the specified path. If DCMAKE_INSTALL_PREFIX is not specified, it will be installed to the standard path for your system, usually /usr/local/ . -->


<!-- ## Integration in CMake projects

In order to use this library in your CMake project, add the following lines to the CMakeLists.txt of your project:\
(also do not forget to include ViennaHRLE/ViennaLS)

```
set(ViennaPS_DIR "/path/to/your/custom/install/")
find_package(ViennaPS REQUIRED)
add_executable(...)
target_include_directories(${PROJECT_NAME} PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${VIENNAPS_LIBRARIES})
``` -->

### Building examples

The examples can be built using CMake:

```
mkdir build && cd build
cmake .. -DVIENNAPS_BUILD_EXAMPLES=ON
make
```

### Building test

The tests can be built using CMake:

```
mkdir build && cd build
cmake .. -DVIENNAPS_BUILD_TESTS=ON
make
```

## Contributing

If you want to contribute to ViennaPS, make sure to follow the [LLVM Coding guidelines](https://llvm.org/docs/CodingStandards.html). Before creating a pull request, make sure ALL files have been formatted by clang-format, which can be done using the format-project.sh script in the root directory.

## Authors

Current contributors: Tobias Reiter, Josip Bobinac, Xaver Klemenschits

Contact us via: viennatools@iue.tuwien.ac.at

ViennaPS was developed under the aegis of the 'Institute for Microelectronics' at the 'TU Wien'.
http://www.iue.tuwien.ac.at/

License
--------------------------
See file LICENSE in the base directory.
