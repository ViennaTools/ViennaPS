---
layout: default
title: Tutorials
nav_order: 5
has_children: true
permalink: /tutorials/
---

# Tutorials

Welcome to the ViennaPS tutorials section. These guides help you understand the core concepts and practical usage of the library. The tutorials start with basic geometry setup and gradually introduce multi-step patterning flows and physically motivated plasma models.

---

## [1. Introduction & Hello World]({% link tutorials/01_introduction.md %})

Learn what ViennaPS is and how it is structured.
Create a minimal simulation domain and run your first simple process.

You will understand:

* The structure of a ViennaPS script
* How to create and visualize a domain
* How to run a basic process

---

## [2. Geometry Creation]({% link tutorials/02_geometry.md %})

Define simulation domains and construct initial geometries.

Topics include:

* 2D vs. 3D domains
* Grid resolution and boundary conditions
* Built-in primitives such as trenches, holes, and fins
* Exporting surface and volume meshes

---

## [3. Simple Processes]({% link tutorials/03_simple_process.md %})

Understand how surface evolution is performed in ViennaPS.

You will learn:

* How to configure isotropic etching and deposition
* How process duration affects geometry evolution
* How to inspect intermediate and final states

---

## [4. Materials & Masking]({% link tutorials/04_materials.md %})

Work with multiple materials and selective processes.

Topics include:

* Creating new material layers
* Removing materials (e.g., mask stripping)
* Using one material as a hard mask
* Material-dependent behavior in processes

---

## [5. Custom Geometry with ViennaLS]({% link tutorials/05_custom_geometry.md %})

Create more advanced geometries using ViennaLS.

You will learn:

* How to use primitives and Boolean operations
* How to construct composite structures
* How to prepare geometries for process simulation

---

## [6. Emulation: Fin Patterning & Transfer]({% link tutorials/06_emulation.md %})

Simulate a simplified spacer-based pattern transfer flow in 3D.

This tutorial covers:

* Creating a fin mask structure
* Conformal isotropic deposition
* Directional etching to form sidewall spacers
* Mask stripping and final pattern transfer

The example demonstrates how multiple simple process steps can emulate a realistic fabrication sequence.

---

## [7. Single Particle CVD]({% link tutorials/07_single_particle_cvd.md %})

Model deposition using a particle-based transport approach.

Topics include:

* `SingleParticleProcess`
* Sticking probability and angular distribution
* Flux-driven growth in trenches
* Transport limitations in high-aspect-ratio features

This tutorial introduces Monte Carlo-based flux computation in a simple setting.

---

## [8. Multi-Particle Etching]({% link tutorials/08_multi_particle_etch.md %})

Combine multiple particle species in one etching model.

You will learn:

* How to use `MultiParticleProcess`
* How to define custom rate functions
* How to combine neutral and ion fluxes
* How to implement mask selectivity and mask erosion

This provides a flexible framework for building custom plasma-inspired models.

---

## [9. SF₆/O₂ Plasma Etching]({% link tutorials/09_plasma_etching.md %})

Run a physically motivated plasma etching model.

Topics include:

* Setting physical units
* Configuring ion, etchant, and passivation fluxes
* Coverage steady-state initialization
* Advection and ray tracing parameters

This tutorial demonstrates a more realistic plasma etching workflow based on a prebuilt physical model.

---

Together, these tutorials guide you from simple geometry creation to multi-step process emulation and physically motivated plasma etching simulations.
