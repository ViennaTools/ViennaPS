---
layout: default
title: Single Particle CVD Deposition
parent: Tutorials
nav_order: 7
---

# 2D Trench Deposition with `SingleParticleProcess` (CVD)

This tutorial shows a simple **2D** Chemical Vapor Deposition (CVD) example in ViennaPS:

1. Create a 2D simulation domain
2. Generate an initial trench geometry
3. Define a single-particle CVD deposition model
4. Add a new material layer (SiO₂)
5. Run the deposition process
6. Save and visualize results

Download Jupyter Notebook: [07_single_particle_cvd.ipynb](https://raw.githubusercontent.com/ViennaTools/ViennaPS/refs/heads/documentation/examples/tutorials/07_single_particle_cvd.ipynb)

---

## 1. Import ViennaPS

```python
import viennaps as ps  # ViennaPS 2D
```

This imports the ViennaPS Python bindings.
(Here we run a 2D example. No explicit `setDimension` is used.)

---

## 2. Create the 2D Simulation Domain

```python
extent = 30
gridDelta = 0.3

domain = ps.Domain(xExtent=extent, gridDelta=gridDelta)
```

### Parameters

* `xExtent`: lateral size of the simulation window (in x-direction)
* `gridDelta`: grid spacing (controls resolution and runtime)

This sets up a 2D computational domain with a resolution of `0.3`.

---

## 3. Generate the Initial Trench Geometry

We create a rectangular trench that acts as the starting topography.

```python
ps.MakeTrench(domain, trenchWidth=10.0, trenchDepth=30.0).apply()
```

### Parameters

* `trenchWidth = 10` → opening width at the top
* `trenchDepth = 30` → depth of the trench

After this step, the domain contains a trench etched into the substrate.

---

## 4. Save the Initial Geometry

```python
domain.saveVolumeMesh("singleParticleCVD_1")
```

This exports the current geometry to a VTU file.
It is useful for visual inspection and for documenting intermediate steps.

---

## 5. Define the CVD Deposition Model

Now we define a simple single-particle deposition model.

```python
model = ps.SingleParticleProcess(
    rate=1.0,
    stickingProbability=0.1,
    sourceExponent=1.0,
)
```

### Model parameters

* `rate`: base deposition rate on a flat surface
* `stickingProbability`: probability that a particle deposits when it hits the surface

  * smaller values usually increase transport into deep features
* `sourceExponent`: controls the angular distribution of incoming particles

  * `1.0` is a common “isotropic-like” choice in simple examples

This model represents deposition by particles traveling through the trench and sticking on impact.

---

## 6. Add a New Material Layer (SiO₂)

Before deposition, we create a new top-level surface assigned to the deposited material.

```python
domain.duplicateTopLevelSet(ps.Material.SiO2)
```

This tells ViennaPS that the evolving deposited layer is **SiO₂**.
Deposition will then grow this new material on top of the existing geometry.

---

## 7. Run the Deposition Process

```python
processDuration = 10.0
ps.Process(domain, model, processDuration).apply()
```

* `processDuration` is the simulated time interval
* During this time, the geometry evolves as SiO₂ deposits on exposed surfaces
* In a trench, this typically leads to:

  * stronger growth near the opening (more particle access)
  * reduced growth deeper inside (limited transport), depending on sticking

---

## 8. Save and Visualize the Result

```python
domain.saveVolumeMesh("singleParticleCVD_2")
domain.show()
```

* `saveVolumeMesh()` exports the final geometry after deposition
* `show()` visualizes the resulting topography

Download Jupyter Notebook: [07_single_particle_cvd.ipynb](https://raw.githubusercontent.com/ViennaTools/ViennaPS/refs/heads/documentation/examples/tutorials/07_single_particle_cvd.ipynb)
