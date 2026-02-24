---
layout: default
title: Multi Particle Etching
parent: Tutorials
nav_order: 8
---

# 2D Etching with `MultiParticleProcess` (Neutrals + Ions)

This tutorial shows how to use [`MultiParticleProcess`](https://viennatools.github.io/ViennaPS/models/prebuilt/multiParticle.html) in **2D** to model etching driven by multiple particle species.

You will run two cases:

1. Etching with a **single neutral** particle species
2. Etching with **neutral + ion** particle species, including optional mask etching

The key concept is that `MultiParticleProcess` computes **one flux per particle species**, and you provide a **rate function** that converts these fluxes into a local surface velocity (etch or deposition), optionally depending on the material.

Download Jupyter Notebook: [08_multi_particle_etch.ipynb](https://raw.githubusercontent.com/ViennaTools/ViennaPS/refs/heads/documentation/examples/tutorials/08_multi_particle_etch.ipynb)

---

## 1. Import ViennaPS

```python
import viennaps as ps  # ViennaPS 2D
```

---

## 2. Helper: Create a Masked Trench Domain

We define a small helper function so we can reset the geometry between process steps.

```python
def createTrenchMask():
    extent = 30
    gridDelta = 0.3

    domain = ps.Domain(xExtent=extent, gridDelta=gridDelta)

    # trenchDepth=0 creates a flat bottom surface, maskHeight adds a mask on top
    ps.MakeTrench(domain, trenchWidth=10.0, trenchDepth=0.0, maskHeight=2.0).apply()

    return domain
```

### What this creates

* A 2D domain of width `30` with resolution `0.3`
* A trench opening of width `10`
* No etched depth in the substrate (`trenchDepth=0.0`), but a **mask layer** of height `2.0`

This is a simple setup to demonstrate selective etching and mask interaction.

---

# Step 1: Etching with a Single Neutral Particle

## 3. Initialize Domain and Save Initial State

```python
domain = createTrenchMask()
domain.saveVolumeMesh("multiParticleEtching_1")
```

This writes the starting geometry to a VTU file for later comparison.

---

## 4. Create the Multi-Particle Model and Add a Neutral Species

```python
model = ps.MultiParticleProcess()

# one neutral particle species
model.addNeutralParticle(stickingProbability=0.2)
```

### Meaning

* `addNeutralParticle(...)` adds a neutral particle source
* `stickingProbability=0.2` controls how often neutrals stick/react at impact
* With only one species, the flux array will contain:

  * `fluxes[0]` → neutral flux

---

## 5. Define the Etch Rate Function

`MultiParticleProcess` needs a function that maps local fluxes to a local surface rate.

```python
neutralRate = 1.0

def rateFunction(fluxes, material):
    if material == ps.Material.Mask:
        return 0  # no etching of mask

    return -neutralRate * fluxes[0]

model.setRateFunction(rateFunction)
```

### Notes

* The function is called for each surface point during the process.
* `material` lets you implement selectivity (here: mask is protected).
* The negative sign means **etching** (surface moves inward).
* For this single-neutral case:

  * Only `fluxes[0]` exists (neutral flux).

---

## 6. Run the Process and Save Result

```python
processDuration = 5.0
ps.Process(domain, model, processDuration).apply()
domain.saveVolumeMesh("multiParticleEtching_2")
```

You now have an etched profile driven only by neutral transport, with the mask unaffected.

---

# Step 2: Etching with Neutral + Ion Particles

## 7. Reset Domain

```python
domain = createTrenchMask()
```

This restores the original geometry so the second case starts from the same initial state.

---

## 8. Create a Model with Neutral and Ion Species

```python
model = ps.MultiParticleProcess()
model.addNeutralParticle(stickingProbability=0.2)

# add an ion species with a directional angular distribution
model.addIonParticle(sourcePower=500.0, thetaRMin=60.0, thetaRMax=90.0)
```

### Meaning

* This time the flux array contains two entries:

  * `fluxes[0]` → neutral flux
  * `fluxes[1]` → ion flux
* The ion parameters shape the incoming ion distribution:

  * `sourcePower` affects the ion intensity (model-dependent)
  * `thetaRMin`, `thetaRMax` restrict the angular range (more directional)

---

## 9. Define a Combined Rate Function (Selectivity + Weights)

```python
neutralWeight  = 1.0
ionWeight      = 1.0
maskEtchFactor = 0.1  # 0 = mask fully protected

def rateFunction(fluxes, material):
    neutralFlux = fluxes[0]
    ionFlux = fluxes[1]

    if material == ps.Material.Mask:
        return -maskEtchFactor * ionWeight * ionFlux

    return -(neutralWeight * neutralFlux + ionWeight * ionFlux)

model.setRateFunction(rateFunction)
```

### What this does

* Substrate etch rate depends on **both** neutrals and ions.
* Mask etching is allowed, but reduced:

  * only ions etch the mask
  * scaled by `maskEtchFactor`

This is a typical pattern:

* neutrals contribute more to isotropic components
* ions contribute more to directional components
* mask etch can be tuned separately

---

## 10. Run the Process, Save, and Visualize

```python
processDuration = 5.0
ps.Process(domain, model, processDuration).apply()
domain.saveVolumeMesh("multiParticleEtching_3")

domain.show()
```

At this point you can compare:

* `multiParticleEtching_2` (neutral-only etch)
* `multiParticleEtching_3` (neutral + ion, with optional mask erosion)

Download Jupyter Notebook: [08_multi_particle_etch.ipynb](https://raw.githubusercontent.com/ViennaTools/ViennaPS/refs/heads/documentation/examples/tutorials/08_multi_particle_etch.ipynb)

