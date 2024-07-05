---
layout: default
title: Custom Models
parent: Process Models
nav_order: 0
has_children: true
---

# Custom Models
{: .fs-9 .fw-500}

---

Users can create a custom process model by interfacing one or more of the classes described below and then inserting them into their custom process model.

- The `SurfaceModel` class is used to describe surface reactions, combining particle fluxes with the surface chemical reactions.
- The `VelocityField` provides the interface between surface velocities and the advection kernel to integrate the Level-Set equation in a time step.
- ViennaPS seamlessly integrates ViennaRay functionality through the `viennaray::Particle` class, providing users with a versatile interface to define the key characteristics of the simulated particle species.

