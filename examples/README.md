# ViennaPS Examples

[Back to the main README](../README.md)

## Building

Run the following commands from the repository root to build the C++ examples:

```bash
cmake -B build -DVIENNAPS_BUILD_EXAMPLES=ON
cmake --build build
```

The examples can then be executed in their respective build folders with the config files, e.g.:
```bash
cd build/examples/exampleName
./exampleName.bat config.txt # (Windows)
./exampleName config.txt # (Other)
```

Individual examples can also be built by calling `make` in their respective build folder. Many examples also include Python scripts using the ViennaPS bindings.

## Trench Deposition

This [example](./trenchDeposition) simulates particle deposition in a trench. It defaults to 3D; set `D = 2` in `trenchDeposition.cpp` for 2D. Adjust process and geometry parameters in `config.txt`. The image compares different particle sticking probabilities *s*.

<div align="center">
  <img src="../assets/deposition.png" width=700 style="background-color:white;">
</div>

## SF6/O2 Hole Etching

This [example](./holeEtching) simulates SF<sub>6</sub>/O<sub>2</sub> plasma etching with ion bombardment. Geometry and plasma conditions are configured in `config.txt`. The image compares hole profiles for different ion and neutral fluxes, as tested in `testFluxes.py`.

> [!NOTE] 
> The underlying model may change in future releases, so running this example in newer versions of ViennaPS might not always reproduce exactly the same results.  
> The images shown here were generated using **ViennaPS v3.6.0**.

<div align="center">
  <img src="../assets/sf6o2_results.png" width=700 style="background-color:white;">
</div>

## Bosch Process

This [example](./boschProcess) compares three approaches to simulating the Bosch deep reactive ion etching (DRIE) process. From left to right, the image shows process emulation, a simple simulation model, and a more detailed physical model.

<div align="center">
  <img src="../assets/bosch_process.png" width=700 style="background-color:white;">
</div>

## Wet Etching

This [example](./cantileverWetEtching) demonstrates the wet etching process, specifically focusing on the cantilever structure. The simulation captures the etching dynamics and the influence of crystallographic directions on the etch profile.

<div align="center">
  <img src="../assets/wet_etching.png" width=700 style="background-color:white;">
</div>

## Selective Epitaxy

This [example](./selectiveEpitaxy) demonstrates the selective epitaxy process, focusing on the growth of SiGe on a Si substrate. Similar to wet etching, the process is influenced by crystallographic directions, which can be adjusted in the __config.txt__ file. The simulation captures the growth dynamics and the resulting SiGe structure.

<div align="center">
  <img src="../assets/epitaxy.png" width=700 style="background-color:white;">
</div>

## Redeposition During Selective Etching

This [example](./oxideRegrowth) models byproduct transport and redeposition during selective etching of a Si<sub>3</sub>N<sub>4</sub>/SiO<sub>2</sub> stack. A convection-diffusion equation tracks byproducts in the etching solution, and their accumulation determines surface regrowth.

<div align="center">
  <img src="../assets/redeposition.gif" width=700 style="background-color:white;">
</div>

## GDS Mask Import Example

This [example](./GDSReader) tests the full GDS mask import, blurring, rotation, scaling, and flipping as well as the level set conversion pipeline. Shown below is the result after applying proximity correction and extrusion on a simple test.

<div align="center">
  <img src="../assets/masks.png" width=1200 style="background-color:white;">
</div>

## Fin Oxidation

This [example](./finOxidation) simulates thermal oxidation of a silicon fin, capturing non-uniform oxide growth and corner rounding due to crystallographic anisotropy. The image shows the initial fin on the left and the oxidized structure with its pressure field on the right.

<div align="center">
  <img src="../assets/fin_oxidation.png" width=700 style="background-color:white;">
</div>

## LOCOS Oxidation

This [example](./locosOxidation) simulates Local Oxidation of Silicon (LOCOS), including the characteristic **bird's beak** beneath the nitride mask. The model couples oxidant diffusion, viscous oxide flow, and mask bending. The image shows the oxidized material stack on the left and nitride stress and oxide pressure on the right.

<div align="center">
  <img src="../assets/locos.png" width=700 style="background-color:white;">
</div>
