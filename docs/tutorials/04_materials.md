---
layout: default
title: Materials & Masking
parent: Tutorials
nav_order: 4
---

# Materials & Masking

Real fabrication involves multiple materials (Silicon, Oxide, Photoresist, etc.). ViennaPS tracks which material defines which part of the volume using a stack of level sets.

## Material-Selective Processes

You can define process rates that depend on the material. This is crucial for masks (which shouldn't etch) or selective deposition.

```python
# Define specific rates for different materials
rates = {
    ps.Material.Si:   -10.0,  # Etch Silicon fast
    ps.Material.Mask:  -0.1,  # Etch Mask very slowly (selectivity 100:1)
    ps.Material.SiO2:   0.0   # Do not etch Oxide at all
}

# Create model with material-specific dictionary
etch_model = ps.IsotropicProcess(materialRates=rates)

# Apply for 5 time units
ps.Process(domain, etch_model, 5.0).apply()
```

## Creating Multi-Material Geometries

ViennaPS uses a **stack of level sets** where each level set represents a material interface. The order matters: newer materials are on top.

### Pattern 1: Duplication
The most common pattern for deposition is to duplicate the top surface:

```python
# 1. Create Silicon substrate
ps.MakePlane(domain, height=0.0, material=ps.Material.Si).apply()

# 2. Duplicate the top surface to define a new layer (e.g. deposited oxide)
domain.duplicateTopLevelSet(ps.Material.SiO2)

# 3. Grow the oxide layer
ps.Process(domain, ps.IsotropicProcess(rate=1.0), 10.0).apply()
```

### Pattern 2: Removing Materials
After processing, you can remove materials (e.g., stripping a mask):

```python
# Remove all level sets with the Mask material
domain.removeMaterial(ps.Material.Mask)

# Or remove just the topmost level set
domain.removeTopLevelSet()
```

### Pattern 3: Mask Material in Process Models
Many process models accept a `maskMaterial` parameter to exclude certain materials:

```python
# Directional etch that doesn't affect the Mask material
model = ps.DirectionalProcess(
    direction=[0.0, 1.0],
    directionalVelocity=-1.0,
    isotropicVelocity=0.0,
    maskMaterial=ps.Material.Mask
)
```
