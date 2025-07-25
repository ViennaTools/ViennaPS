---
layout: default
title: Examples
nav_order: 13
---

<style>
  table img {
    width: 300px;
    height: 250px;
    object-fit: cover;
  }
</style>

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

## Examples

| -------------------------------------------------- | -------------------------------------------------- |
| **Atomic Layer Deposition**<br/>[![]({% link assets/examples/ALD.png %})](https://github.com/ViennaTools/ViennaPS/tree/master/examples/atomicLayerDeposition) | **Blazed Gratings Etching**<br/>[![]({% link assets/examples/blazedGrating.png %})](https://github.com/ViennaTools/ViennaPS/tree/master/examples/blazedGratingsEtching) |
| **Bosch Process**<br/>[![]({% link assets/examples/BoschProcess.png %})](https://github.com/ViennaTools/ViennaPS/tree/master/examples/boschProcess)  | **Cantilever Wet Etching**<br/>[![]({% link assets/examples/wetEtching.png %})](https://github.com/ViennaTools/ViennaPS/tree/master/examples/cantileverWetEtching)       |
| **Emulation**<br/>[![]({% link assets/examples/FinFET.png %})](https://github.com/ViennaTools/ViennaPS/tree/master/examples/emulation) | **DRAM Wiggling**<br/>[![]({% link assets/examples/DRAM.png %})](https://github.com/ViennaTools/ViennaPS/tree/master/examples/DRAMWiggling) |
| **Faraday Cage Etching**<br/>[![]({% link assets/examples/faradayCageEtching.png %})](https://github.com/ViennaTools/ViennaPS/tree/master/examples/faradayCageEtching) | **Hole Etching**<br/>[![]({% link assets/examples/holeEtching.png %})](https://github.com/ViennaTools/ViennaPS/tree/master/examples/holeEtching) |
| **Ion Beam Etching**<br/>[![]({% link assets/examples/IBE.png %})](https://github.com/ViennaTools/ViennaPS/tree/master/examples/ionBeamEtching) | **Selective Epitaxy**<br/>[![]({% link assets/examples/epitaxy.png %})](https://github.com/ViennaTools/ViennaPS/tree/master/examples/selectiveEpitaxy) |
| **Sputter Deposition**<br/>[![]({% link assets/examples/sputterDepo.png %})](https://github.com/ViennaTools/ViennaPS/tree/master/examples/sputterDeposition) | **Stack Etching**<br/>[![]({% link assets/examples/stackEtching.png %})](https://github.com/ViennaTools/ViennaPS/tree/master/examples/stackEtching) |
| **TEOS Trench Deposition**<br/>[![]({% link assets/examples/TEOS.png %})](https://github.com/ViennaTools/ViennaPS/tree/master/examples/TEOSTrenchDeposition) | **Trench Deposition**<br/>[![]({% link assets/examples/trenchDepo.png %})](https://github.com/ViennaTools/ViennaPS/tree/master/examples/trenchDeposition) |

More Examples:
* [Custom Example Process](https://github.com/ViennaTools/ViennaPS/tree/master/examples/exampleProcess)
* [GDS Reader](https://github.com/ViennaTools/ViennaPS/tree/master/examples/GDSReader)
* [Interpolation Demo](https://github.com/ViennaTools/ViennaPS/tree/master/examples/interpolationDemo)
* [Oxide Regrowth](https://github.com/ViennaTools/ViennaPS/tree/master/examples/oxideRegrowth)
* [SiGe Selective Etching](https://github.com/ViennaTools/ViennaPS/tree/master/examples/SiGeSelectiveEtching)
* [Trench Deposition Geometric](https://github.com/ViennaTools/ViennaPS/tree/master/examples/trenchDepositionGeometric)