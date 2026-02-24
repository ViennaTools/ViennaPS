---
layout: default
title: Process Emulation
parent: Tutorials
nav_order: 6
---

# Fin Patterning and Sidewall Transfer (3D)

This tutorial demonstrates a simple 3D process flow using ViennaPS:

1. Create a simulation domain
2. Add a mask feature (fin)
3. Perform isotropic deposition
4. Perform directional etching
5. Strip the mask
6. Transfer the pattern into the substrate

The example mimics a simplified sidewall transfer process.

Download Jupyter Notebook: [06_emulation.ipynb](https://raw.githubusercontent.com/ViennaTools/ViennaPS/refs/heads/documentation/examples/tutorials/06_emulation.ipynb)

---

## 1. Import ViennaPS and Set Dimension

```python
import viennaps as ps 
ps.setDimension(3)
```

ViennaPS supports 2D and 3D simulations.
Here we explicitly set the simulation to **3D**.

---

## 2. Create the Simulation Domain

```python
extent = 30
gridDelta = 0.3

domain = ps.Domain(
    xExtent=extent,
    yExtent=extent,
    gridDelta=gridDelta,
    boundary=ps.BoundaryType.REFLECTIVE_BOUNDARY
)
```

### Parameters

* `xExtent`, `yExtent`: Lateral size of the simulation window
* `gridDelta`: Grid spacing (controls resolution)
* `boundary`: Reflective boundaries prevent material from leaving the domain

This creates a square domain of size 30 × 30 units with a resolution of 0.3 units.

---

## 3. Create a Fin Mask Structure

We now add a rectangular fin structure that acts as a mask.

```python
ps.MakeFin(
    domain,
    finWidth=6.,
    finHeight=0.,
    maskHeight=10.0
).apply()
```

### Parameters

* `finWidth = 6` → width of the feature
* `finHeight = 0` → no substrate protrusion (mask only)
* `maskHeight = 10` → height of the resist layer

This creates a vertical resist structure on top of the substrate.

---

## 4. Save and Visualize the Initial Geometry

```python
domain.saveSurfaceMesh("emulation_1", addInterfaces=True)
domain.show()
```

* `saveSurfaceMesh()` exports the current geometry
* `addInterfaces=True` includes material interfaces
* `show()` opens the built-in viewer

At this stage, only the mask structure is present.

---

## 5. Isotropic Deposition (Conformal Coating)

Next, we deposit a conformal SiO₂ layer.

### Duplicate the Top Level Set

```python
domain.duplicateTopLevelSet(ps.Material.SiO2)
```

This creates a new material layer (SiO₂) on top of the existing geometry.

### Apply Isotropic Deposition

```python
isoDepo = ps.IsotropicProcess(rate=1.0)
ps.Process(domain, isoDepo, 4.0).apply()
```

### Explanation

* `rate = 1.0` → constant deposition rate
* `time = 4.0` → process duration

Since the process is isotropic, deposition occurs uniformly in all directions.
This produces a conformal coating over the mask and substrate.

```python
domain.show()
```

You should now see a conformal SiO₂ layer.

---

## 6. Directional Etch (Top Removal)

Now we remove the horizontal parts of the deposited layer.

```python
directionalEtch = ps.DirectionalProcess(
    direction=[0., 0., 1.],
    directionalVelocity=1.0,
    isotropicVelocity=0.0,
    maskMaterial=ps.Material.Undefined,
    calculateVisibility=False
)

ps.Process(domain, directionalEtch, 5.0).apply()
```

### Parameters

* `direction=[0,0,1]` → etch in vertical direction
* `directionalVelocity=1.0` → pure anisotropic etch
* `isotropicVelocity=0.0` → no lateral component
* `calculateVisibility=False` → no shadowing

This step removes the horizontal SiO₂ but leaves material on the sidewalls.

```python
domain.show()
```

You should now observe SiO₂ spacers along the mask sidewalls.

---

## 7. Remove the Original Mask

```python
domain.removeMaterial(ps.Material.Mask)
```

This simulates resist stripping.
Only the SiO₂ sidewall spacers remain.

---

## 8. Final Directional Etch (Pattern Transfer)

We now etch the substrate using the SiO₂ spacers as a hard mask.

```python
directionalEtch = ps.DirectionalProcess(
    direction=[0., 0., 1.],
    directionalVelocity=1.0,
    isotropicVelocity=0.0,
    maskMaterial=ps.Material.SiO2,
    calculateVisibility=False
)

ps.Process(domain, directionalEtch, 20.0).apply()
```

### Key Difference

* `maskMaterial=ps.Material.SiO2`

This means the SiO₂ spacers are protected and act as a hard mask.

After 20 time units, the substrate is etched vertically beneath the spacers.

```python
domain.saveVolumeMesh("emulation_5")
domain.show()
```

The final structure represents a transferred fin pattern defined by the spacer width.

Download Jupyter Notebook: [06_emulation.ipynb](https://raw.githubusercontent.com/ViennaTools/ViennaPS/refs/heads/documentation/examples/tutorials/06_emulation.ipynb)
