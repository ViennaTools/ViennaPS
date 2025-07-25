---
layout: default
title: Velocity Field
parent: Custom Models
grand_parent: Process Models
nav_order: 2
---

# Velocity Field
{: .fs-9 .fw-500}

---

Extends the surface reaction rates to all grid points in the level-set representation. The default velocity extension uses the closest surface grid point for each level-set grid point. This should be used if a surface rate model, which depends on surface fluxes or other surface properties, is defined. If the model is analytic, the rates can be applied directly in the velocity field for each grid point.  

## Options


| Option | Description                                      |
|--------|--------------------------------------------------|
|   0    | Do not translate level set ID to surface ID. This should be enabled if the surface velocity is only provided in the `VelocityField` class and not through the `SurfaceModel` class.     |
|   1    | Use unordered map to translate level set ID to surface ID. |
|   2    | (Default) Use KD-tree to translate level set ID to surface ID. The KD-tree uses a nearest neighbor lookup to determine the closest surface point and according velocity.  |
