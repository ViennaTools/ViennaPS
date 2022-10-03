# ViennaPS

ViennaPS is a header-only C++ process simulation library, which includes surface and volume representations, a ray tracer, and physical models for the simulation of microelectronic fabrication processes. 

IMPORTANT NOTE: ViennaPS is under heavy development and improved daily. If you do have suggestions or find bugs, please let us know!

## Releases
Releases are tagged on the master branch and available in the [releases section](https://github.com/ViennaTools/ViennaPS/releases).

## Building

### Supported Operating Systems

* Windows (Visual Studio)

* Linux (g++ / clang)

* macOS (XCode)


### System Requirements

* C++17 Compiler with OpenMP support

### Dependencies

* [ViennaLS](https://github.com/ViennaTools/viennals)

* [ViennaRay](https://github.com/ViennaTools/viennaray)

The dependencies will be installed automatically, if no system wide installation is found and no paths are given during configuration of the project. 
<!-- ## Using ViennaPS in your project

Have a look at the [example repo](https://github.com/ViennaTools/viennals-example) for creating a project with ViennaPS as a dependency. -->


## Installing

Since this is a header only project, it does not require any installation.
However, we recommend the following procedure.

<!-- Make sure you have [ViennaLS](https://github.com/ViennaTools/viennals) and [ViennaRay](https://github.com/ViennaTools/ViennaRay) installed on your system and run: -->

```
git clone github.com/ViennaTools/ViennaPS.git
cd ViennaPS
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/your/custom/install/
make buildDependencies # this will install the required dependencies the first time it is called and might take a while
make install
```

This will install the necessary headers and CMake files to the specified path. If `CMAKE_INSTALL_PREFIX` is not specified, it will be installed to the standard path for your system, usually `/usr/local/` . 

If you want to use your own install of [ViennaLS](https://github.com/ViennaTools/viennals) and [ViennaRay](https://github.com/ViennaTools/viennaray), just specify the directory in CMake: 

```
git clone github.com/ViennaTools/ViennaPS.git
cd ViennaPS
mkdir build && cd build
cmake .. -DViennaLS_DIR=path/to/ViennaLS/install/ -DViennaRay_DIR=path/to/ViennaRay/install/
make install
```

## Integration in CMake projects

In order to use this library in your CMake project, add the following lines to the CMakeLists.txt of your project:

```
set(ViennaPS_DIR "/path/to/your/custom/install/")
find_package(ViennaPS REQUIRED)
add_executable(...)
target_include_directories(${PROJECT_NAME} PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${VIENNAPS_LIBRARIES})
``` 

### Building examples

The examples can be built using CMake. Make sure you have the dependencies installed and run:

```
mkdir build && cd build
cmake .. -DVIENNAPS_BUILD_EXAMPLES=ON
make buildExamples
```

<!-- ### Building test

The tests can be built using CMake:

```
mkdir build && cd build
cmake .. -DVIENNAPS_BUILD_TESTS=ON
make
``` -->

## Support

Basic [Examples](https://github.com/ViennaTools/ViennaPS/tree/master/Examples) can be found online. 

Bug reports and suggestions should be filed on GitHub.

## Contributing

If you want to contribute to ViennaPS, make sure to follow the [LLVM Coding guidelines](https://llvm.org/docs/CodingStandards.html). Before creating a pull request, make sure ALL files have been formatted by clang-format, which can be done using the format-project.sh script in the root directory.

## Authors

Current contributors: Tobias Reiter, Josip Bobinac, Xaver Klemenschits, Julius Piso

Contact us via: viennatools@iue.tuwien.ac.at

ViennaPS was developed under the aegis of the 'Institute for Microelectronics' at the 'TU Wien'.
http://www.iue.tuwien.ac.at/

License
--------------------------
See file LICENSE in the base directory.
