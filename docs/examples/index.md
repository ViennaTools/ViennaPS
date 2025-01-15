---
layout: default
title: Examples
nav_order: 13
---

# Examples
{: .fs-9 .fw-700}

---

## Building

The examples can be built using CMake:

```bash
git clone https://github.com/ViennaTools/ViennaPS.git
cd ViennaPS

cmake -B build -DVIENNAPS_BUILD_EXAMPLES=ON
cmake --build build
```

The examples can then be executed in their respective build folders with the config files, e.g.:
```bash
cd examples/exampleName
./ExampleName.bat config.txt # (Windows)
./ExampleName config.txt # (Other)
```

Individual examples can also be build by calling `make` in their respective build folder. An equivalent Python script, using the ViennaPS Python bindings, is also given for each example. 

## Examples List

* [Atomic Layer Deposition](https://github.com/ViennaTools/ViennaPS/tree/master/examples/atomicLayerDeposition)
* [Bosch Process](https://github.com/ViennaTools/ViennaPS/tree/master/examples/boschProcess)
* [Custom Example Process](https://github.com/ViennaTools/ViennaPS/tree/master/examples/exampleProcess)
* [GDS Reader](https://github.com/ViennaTools/ViennaPS/tree/master/examples/GDSReader)
* [Cantilever Wet Etching](https://github.com/ViennaTools/ViennaPS/tree/master/examples/cantileverWetEtching)
* [Selective Epitaxy](https://github.com/ViennaTools/ViennaPS/tree/master/examples/selectiveEpitaxy)
* [Trench Deposition](https://github.com/ViennaTools/ViennaPS/tree/master/examples/trenchDeposition)
* [Trench Deposition Geometric](https://github.com/ViennaTools/ViennaPS/tree/master/examples/trenchDepositionGeometric)
* [TEOS Trench Deposition](https://github.com/ViennaTools/ViennaPS/tree/master/examples/TEOSTrenchDeposition)
* [Hole Etching](https://github.com/ViennaTools/ViennaPS/tree/master/examples/holeEtching)
* [Stack Etching](https://github.com/ViennaTools/ViennaPS/tree/master/examples/stackEtching)
* [Oxide Regrowth](https://github.com/ViennaTools/ViennaPS/tree/master/examples/oxideRegrowth)
* [Interpolation Demo](https://github.com/ViennaTools/ViennaPS/tree/master/examples/interpolationDemo)
