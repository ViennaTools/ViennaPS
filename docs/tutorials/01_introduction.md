---
layout: default
title: Introduction
parent: Tutorials
nav_order: 1
---

# Introduction to ViennaPS

ViennaPS is a header-only C++ library (with Python bindings) for topography simulation in microelectronic fabrication. It combines:
*   **Level Set Method** (via [ViennaLS](https://github.com/ViennaTools/viennals)): To track the moving surface interface implicitly.
*   **Monte Carlo Ray Tracing** (via [ViennaRay](https://github.com/ViennaTools/viennaray)): To calculate particle fluxes from the source to the surface.

## Your First Simulation

A minimal ViennaPS script consists of three parts:
1.  **Setup**: Import library and set dimension.
2.  **Geometry**: Define the initial domain.
3.  **Process**: Define and apply a process model.

### Hello World Example

Here is a simple script that creates a flat surface and grows a layer on top of it.

```python
import viennaps as ps

# 1. Setup
# Set simulation dimension to 2D
ps.setDimension(2)

# 2. Geometry
# Create a domain with grid spacing 1.0 and width 100
domain = ps.Domain(gridDelta=1.0, xExtent=100.0)

# Create a flat substrate of Silicon (Si) at height 0
ps.MakePlane(domain, height=0.0, material=ps.Material.Si).apply()

# Save the initial state
domain.saveSurfaceMesh("hello_world_initial")

# 3. Process
# Define an isotropic deposition model with rate 1.0
model = ps.IsotropicProcess(rate=1.0)

# Create and run the process for 10 time units
process = ps.Process(domain, model, 10.0)
process.apply()

# Save the final state
domain.saveVolumeMesh("hello_world_final")
```

Save this code as `hello_world.py` and run it with `python hello_world.py`. You will generate `.vtu` files that can be viewed in [ParaView](https://www.paraview.org/), or for a quick preview, you can use the built-in viewer in ViennaPS: `domain.show()`.

{: .note }
ViennaPS does not enforce specific units. It is common to use **nanometers** for length and **seconds** for time, but you can choose any consistent unit system.
