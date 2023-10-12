# ViennaPS
[![Linux](https://github.com/ViennaTools/ViennaPS/actions/workflows/linux_test.yml/badge.svg)](https://github.com/ViennaTools/ViennaPS/actions/workflows/linux_test.yml)
[![macOS](https://github.com/ViennaTools/ViennaPS/actions/workflows/macos_test.yml/badge.svg)](https://github.com/ViennaTools/ViennaPS/actions/workflows/macos_test.yml)
[![Windows](https://github.com/ViennaTools/ViennaPS/actions/workflows/windows_test.yml/badge.svg)](https://github.com/ViennaTools/ViennaPS/actions/workflows/windows_test.yml)

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

### Dependencies (installed automatically)

* [ViennaLS](https://github.com/ViennaTools/viennals)

* [ViennaRay](https://github.com/ViennaTools/viennaray)

## Installing

The CMake configuration automatically checks if the dependencies are installed. If CMake is unable to find them, the dependencies will be built from source with the _buildDependencies_ target.

```bash
git clone https://github.com/ViennaTools/ViennaPS.git
cd ViennaPS
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/your/custom/install/
make buildDependencies # this will install all dependencies and might take a while
make install
```

This will install the necessary headers and CMake files to the specified path. If `CMAKE_INSTALL_PREFIX` is not specified, it will be installed to the standard path for your system, usually `/usr/local/` . 

If one wants to use a specific installation of one or more of the dependencies, just pass the corresponding _*_DIR_ variable as a configuration option (e.g. -DViennaLS_DIR=/path/to/viennals/install -DViennaRay_DIR=/path/to/viennaray/install)

## Integration in CMake projects

In order to use this library in your CMake project, add the following lines to the CMakeLists.txt of your project:

```CMake
set(ViennaPS_DIR "/path/to/your/custom/install/")
find_package(ViennaPS REQUIRED)
add_executable(${PROJECT_NAME} ...)
target_include_directories(${PROJECT_NAME} PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${VIENNAPS_LIBRARIES})
``` 

## Basic Examples

### Building

The examples can be built using CMake:
> __Important__: Make sure all dependencies are installed and have been built previously

```bash
mkdir build && cd build
cmake .. -DVIENNAPS_BUILD_EXAMPLES=ON
make buildExamples
```

The examples can then be executed in their respective build folders with the config files, e.g.:
```bash
cd Examples/ExampleName
./ExampleName config.txt
```

Individual examples can also be build by calling `make` in their respective build folder.

### Trench Deposition

This example contains a single particle deposition process in a trench geometry. By default, a 2D representation of the trench is simulated. However, 3D simulations are also possible by setting the value of the constant _D_ in __TrenchDeposition.cpp__ to 3. In the __config.txt__ file the process and geometry parameters can be varied. 
The picture show an example of the trench deposition process for various value of the particle sticking probability _s_.
<div align="center">
  <img src="https://raw.githubusercontent.com/ViennaTools/ViennaPS/master/data/images/deposition.svg" width=700 style="background-color:white;">
</div>

### SF<sub>6</sub>O<sub>2</sub> Hole Etching

This example demonstrates a hole etching process with a SF<sub>6</sub>O<sub>2</sub> plasma etching chemistry with ion bombardement. The process and geometry parameters can be varied in the __config.txt__ file. 
Below the results after 1, 2, and 3 seconds of etching are shown.
<div align="center">
  <img src="https://raw.githubusercontent.com/ViennaTools/ViennaPS/master/data/images/hole_etching.svg" width=700 style="background-color:white;">
</div>

By changing the dimension of the hole etching example (_D = 2_), we can easily simulate the profile of a trench etching process with the same plasma chemistry. Here we can, for example, vary the mask tapering angle to observe increased microtrenching, as shown below.
<div align="center">
  <img src="https://raw.githubusercontent.com/ViennaTools/ViennaPS/master/data/images/sidewall_tapering.svg" width=700 style="background-color:white;">
</div>

### Redeposition During Selective Etching

This example demonstrates capturing etching byproducts and the subsequent redeposition during a selective etching process in a Si<sub>3</sub>N<sub>4</sub>/SiO<sub>2</sub> stack. The etching byproducts are captured in a cell set description of the etching plasma. To model the dynamics of these etching byproducts, a convection-diffusion equation is solved on the cell set using finite differences. The redeposition is then captured by adding up the byproducts in every step and using this information to generate a velocity field on the etched surface. 
<div align="center">
  <img src="https://raw.githubusercontent.com/ViennaTools/ViennaPS/master/data/images/redeposition.gif" width=700 style="background-color:white;">
</div>

## Application

It is also possible to build an application which can parse a custom configuration file and execute pre-defined processes. The application can be built using CMake:
> __Important__: Make sure all dependencies are installed and have been built previously
```bash
mkdir build && cd build
cmake .. -DVIENNAPS_BUILD_APPLICATION=ON
make buildApplication
```
This creates 2 executables `ViennaPS2D` and `ViennaPS3D` which run processes in 2 or 3 dimensions respectively. Every configuration file can be run in 2D or 3D mode.

The configuration file must obey a certain structure in order to be parsed correctly. An example for a configuration file can be seen in _SampleConfig.txt_. The configuration file is parsed line by line and each succesfully parsed line is executed immediately. A detailed documentation for the configuration file can be found in **app/README.md**.


## Contributing

If you want to contribute to ViennaPS, make sure to follow the [LLVM Coding guidelines](https://llvm.org/docs/CodingStandards.html). Before creating a pull request, make sure ALL files have been formatted by clang-format, which can be done using the format-project.sh script in the root directory.

## Authors

Current contributors: Tobias Reiter, Julius Piso, Josip Bobinac, Xaver Klemenschits

Contact us via: viennatools@iue.tuwien.ac.at

ViennaPS was developed under the aegis of the 'Institute for Microelectronics' at the 'TU Wien'.
http://www.iue.tuwien.ac.at/

License
--------------------------
See file LICENSE in the base directory.
