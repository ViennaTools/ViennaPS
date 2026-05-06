---
layout: default
title: Pre-Built Models
parent: Process Models
nav_order: 1
has_children: true
---

# Pre-Built Models
{: .fs-9 .fw-500}

ViennaPS provides ready-to-use process models for common etching, deposition, and process-emulation workflows.

## Documented Models

* [Isotropic Process]({% link models/prebuilt/isotropic.md %})
* [Directional Process]({% link models/prebuilt/directional.md %})
* [Single Particle Process]({% link models/prebuilt/singleParticle.md %})
* [Multi Particle Process]({% link models/prebuilt/multiParticle.md %})
* [SF<sub>6</sub>/O<sub>2</sub> Etching]({% link models/prebuilt/SF6O2Etching.md %})
* [TEOS Deposition]({% link models/prebuilt/TEOS.md %})
* [TEOS PE CVD Process]({% link models/prebuilt/TEOSPECVD.md %})
* [Fluorocarbon Etching]({% link models/prebuilt/fluorocarbonEtching.md %})
* [Ion Beam Etching]({% link models/prebuilt/IBE.md %})
* [Wet Etching]({% link models/prebuilt/wetEtching.md %})
* [Selective Epitaxy]({% link models/prebuilt/selectiveEpitaxy.md %})
* [Geometric Models]({% link models/prebuilt/geometric.md %})
* [Oxide Regrowth]({% link models/prebuilt/oxideRegrowth.md %})

## Exposed Models With Example Coverage

The following pre-built models are exposed in the C++ and Python APIs, but currently have example-based coverage rather than dedicated reference pages:

| Model | Related example |
|-------|-----------------|
| `CF4O2Etching` | [SiGe Selective Etching](https://github.com/ViennaTools/ViennaPS/tree/master/examples/SiGeSelectiveEtching) |
| `HBrO2Etching` | [DRAM Wiggling](https://github.com/ViennaTools/ViennaPS/tree/master/examples/DRAMWiggling) |
| `FaradayCageEtching` | [Faraday Cage Etching](https://github.com/ViennaTools/ViennaPS/tree/master/examples/faradayCageEtching) |
| `CSVFileProcess` | [Sputter Deposition](https://github.com/ViennaTools/ViennaPS/tree/master/examples/sputterDeposition) |
| `SingleParticleALD` | [Atomic Layer Deposition](https://github.com/ViennaTools/ViennaPS/tree/master/examples/atomicLayerDeposition) |
| `SF6C4F8Etching` | [Bosch Process](https://github.com/ViennaTools/ViennaPS/tree/master/examples/boschProcess) |
| `GeometricTrenchDeposition` | [Trench Deposition Geometric](https://github.com/ViennaTools/ViennaPS/tree/master/examples/trenchDepositionGeometric) |
