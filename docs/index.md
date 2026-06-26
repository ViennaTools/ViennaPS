---
title: Home
layout: default
nav_order: 1
---

<!-- # ViennaPS
{: .fs-10 }

Process Simulation Library
{: .fs-6 .fw-300 } -->

![]({% link assets/images/banner.png %})

[Get started now]({% link inst/index.md %}){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View it on GitHub](https://github.com/ViennaTools/ViennaPS){: .btn .fs-5 .mb-4 .mb-md-0 }

---

ViennaPS is a header-only C++ library for process and topography simulation in
microelectronic fabrication. It models the evolution of 2D and 3D surfaces
during etching, deposition, oxidation, and related steps, combining advanced
level-set methods for surface evolution with Monte Carlo ray tracing for flux
calculation and physics-based solvers for coupled processes.

ViennaPS supports both physics-based process models and fast emulation
approaches, enabling flexible and efficient development of semiconductor
processes. It can be integrated into existing C++ projects and also provides
Python bindings for Python-based workflows.

{: .note }
> ViennaPS is under heavy development and improved daily. If you do have suggestions or find bugs, please let us know on [GitHub][ViennaPS issues] or contact us directly at [viennatools@iue.tuwien.ac.at](mailto:viennatools@iue.tuwien.ac.at)!

This documentation is your guide to installing ViennaPS, building C++ and
Python workflows, selecting process models, and running the provided examples.

---

## Quick Start

Install the Python package from PyPI:

```bash
pip install ViennaPS
```

Then import it in Python:

```python
import viennaps as vps
```

By default, ViennaPS operates in 2D. Use `vps.setDimension(3)` for 3D
workflows.

For C++ projects, ViennaPS is usually consumed with
[CPM.cmake](https://github.com/cpm-cmake/CPM.cmake):

```cmake
CPMAddPackage("gh:viennatools/viennaps@4.6.1")
target_link_libraries(${PROJECT_NAME} PUBLIC ViennaTools::ViennaPS)
```

See [Installing the Library]({% link inst/index.md %}) for full installation
instructions.

## Dependencies

ViennaPS is part of the ViennaTools ecosystem. During CMake configuration, the
required ViennaTools libraries are fetched automatically:

* [ViennaCore](https://github.com/ViennaTools/ViennaCore)
* [ViennaLS](https://github.com/ViennaTools/ViennaLS)
* [ViennaHRLE](https://github.com/ViennaTools/ViennaHRLE)
* [ViennaRay](https://github.com/ViennaTools/ViennaRay)
* [ViennaCS](https://github.com/ViennaTools/ViennaCS)

The main external dependencies are:

* [VTK](https://vtk.org/) 9.0.0 or newer
* [Embree](https://www.embree.org/) 4.0.0 or newer

CMake checks for these dependencies during configuration. If they are not
available, they can be built from source as part of the build. To prefer local
installations, pass their prefixes through `VIENNAPS_LOOKUP_DIRS` or
`CMAKE_PREFIX_PATH`.

## Supported Platforms

ViennaPS supports Linux, macOS, and Windows with a C++20 compiler and OpenMP
support.

## GPU Acceleration

ViennaPS supports experimental GPU acceleration for ray tracing and for the
diffusion solver in the physics-based oxidation model. GPU builds require a
CUDA-capable system. See [Installing the GPU Module]({% link inst/gpu.md %})
for details.

## Tests

ViennaPS uses CTest. To build and run the regular test suite:

```bash
git clone https://github.com/ViennaTools/ViennaPS.git
cd ViennaPS

cmake -B build -DVIENNAPS_BUILD_TESTS=ON
cmake --build build
ctest -E "Benchmark|Performance" --test-dir build
```

---

## Contributing

If you want to contribute to ViennaPS, make sure to follow the [LLVM Coding guidelines](https://llvm.org/docs/CodingStandards.html).

Make sure to format all files before creating a pull request:
```bash
cmake -B build
cmake --build build --target format
```

## About the project

ViennaPS was developed under the aegis of the [Institute for Microelectronics](http://www.iue.tuwien.ac.at/) at the __TU Wien__. 

Current contributors: Tobias Reiter, Noah Karnel, Roman Kostal, Lado Filipovic

Contact us via: [viennatools@iue.tuwien.ac.at](mailto:viennatools@iue.tuwien.ac.at)

## License 

Versions older than 4.3.0 were released under the MIT License. Starting with
version 4.3.0, ViennaPS is licensed under the
[GPL-3.0 License](https://github.com/ViennaTools/ViennaPS/blob/master/LICENSE).

[ViennaPS repo]: https://github.com/ViennaTools/ViennaPS
[ViennaPS issues]: https://github.com/ViennaTools/ViennaPS/issues
