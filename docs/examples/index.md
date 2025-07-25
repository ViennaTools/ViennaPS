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

Individual examples can also be build by calling `make` in their respective build folder. An equivalent Python script, using the ViennaPS Python bindings, is also given for most examples.

## Examples List

* [Atomic Layer Deposition](https://github.com/ViennaTools/ViennaPS/tree/master/examples/atomicLayerDeposition)
* [Blazed Gratings Etchting](https://github.com/ViennaTools/ViennaPS/tree/master/examples/blazedGratingsEtching)
* [Bosch Process](https://github.com/ViennaTools/ViennaPS/tree/master/examples/boschProcess)
* [Custom Example Process](https://github.com/ViennaTools/ViennaPS/tree/master/examples/exampleProcess)
* [Cantilever Wet Etching](https://github.com/ViennaTools/ViennaPS/tree/master/examples/cantileverWetEtching)
* [DRAM Wiggling](https://github.com/ViennaTools/ViennaPS/tree/master/examples/DRAMWiggling)
* [Emulation](https://github.com/ViennaTools/ViennaPS/tree/master/examples/emulation)
* [Faraday Cage Etching](https://github.com/ViennaTools/ViennaPS/tree/master/examples/faradayCageEtching)
* [GDS Reader](https://github.com/ViennaTools/ViennaPS/tree/master/examples/GDSReader)
* [Hole Etching](https://github.com/ViennaTools/ViennaPS/tree/master/examples/holeEtching)
* [Interpolation Demo](https://github.com/ViennaTools/ViennaPS/tree/master/examples/interpolationDemo)
* [Ion Beam Etching](https://github.com/ViennaTools/ViennaPS/tree/master/examples/ionBeamEtching)
* [Oxide Regrowth](https://github.com/ViennaTools/ViennaPS/tree/master/examples/oxideRegrowth)
* [SiGe Selective Etching](https://github.com/ViennaTools/ViennaPS/tree/master/examples/SiGeSelectiveEtching)
* [Selective Epitaxy](https://github.com/ViennaTools/ViennaPS/tree/master/examples/selectiveEpitaxy)
* [Sputter Deposition](https://github.com/ViennaTools/ViennaPS/tree/master/examples/sputterDeposition)
* [Stack Etching](https://github.com/ViennaTools/ViennaPS/tree/master/examples/stackEtching)
* [TEOS Trench Deposition](https://github.com/ViennaTools/ViennaPS/tree/master/examples/TEOSTrenchDeposition)
* [Trench Deposition](https://github.com/ViennaTools/ViennaPS/tree/master/examples/trenchDeposition)
* [Trench Deposition Geometric](https://github.com/ViennaTools/ViennaPS/tree/master/examples/trenchDepositionGeometric)