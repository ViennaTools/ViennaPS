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

ViennaPS is a header-only C++ library for topography simulation in microelectronic fabrication processes. It models the evolution of 2D and 3D surfaces during etching, deposition, and related steps, combining advanced level-set methods for surface evolution with Monte Carlo ray tracing for flux calculation. This allows accurate, feature-scale simulation of complex fabrication geometries.

ViennaPS supports both physical process models and fast emulation approaches, enabling flexible and efficient development of semiconductor processes. It can be easily integrated into existing C++ projects and also provides Python bindings for use in Python-based workflows. The library is actively developed and continuously improved to address the needs of process and topography simulation in microelectronics.

{: .note }
> ViennaPS is under heavy development and improved daily. If you do have suggestions or find bugs, please let us know on [GitHub][ViennaPS issues] or contact us directly at [viennatools@iue.tuwien.ac.at](mailto:viennatools@iue.tuwien.ac.at)!

This documentation is your guide to using and getting the most out of our process simulation library. Whether you're a researcher looking to improve your simulation workflows or an engineer working to optimize fabrication processes, this library offers a flexible and powerful platform to support your work.

Inside, you'll find clear explanations, practical examples, and recommended workflows to help you use the library effectively. Our goal is to give you the knowledge and tools needed to accurately simulate a wide range of fabrication processes, enabling better insights, informed decisions, and innovation in the field.

---

> ⚙️ **ViennaPS v4.0.0 Released — Major Framework Update**
>
> This release introduces a complete rework of the process framework, unified Python bindings, and extended GPU and material support.

## What's New

### Core framework
- Modular **flux engine** with new options: `AUTO` (default), `CPU_DISK`, `GPU_DISK`, `GPU_LINE`, `GPU_TRIANGLE`.
- `AUTO` automatically selects CPU or GPU based on build and model support.
- **AtomicLayerProcess** removed; ALD handled by standard `Process()`.
- New parameter structs:
  - `AtomicLayerProcessParameters`
  - `CoverageParameters`
  - `RayTracingParameters`
  - `AdvectionParameters`
- All parameter structs now use a single `setParameters()` function.

### Python interface
- Unified package: `viennaps` replaces `viennaps2d` and `viennaps3d`.
- Dimension modules available under `viennaps.d2` and `viennaps.d3`.
- Default dimension is 2D; can be changed via `viennaps.setDimension()`.

### Models and simulation
- Extended **material list** with common semiconductor materials.
- **Fluorocarbon model** now supports arbitrary material combinations.
- Fixed issue where **underlying materials** were not etched in geometric models.

### I/O and utilities
- Updated `saveSurfaceMesh()`:
  - Removed `addMaterialIds`
  - Added `addInterfaces` to export all material interfaces.
- Improved extrusion and slicing functions.

### Build system
- GPU builds now **auto-download OptiX headers** if missing.
- Updated CI, OpenMP handling, and dependencies.

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

ViennaPS is licensed under the [MIT License](https://github.com/ViennaTools/ViennaPS/blob/master/LICENSE).

[ViennaPS repo]: https://github.com/ViennaTools/ViennaPS
[ViennaPS issues]: https://github.com/ViennaTools/ViennaPS/issues

