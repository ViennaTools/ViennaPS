---
layout: default
title: Particles - Flux Calculation
parent: Custom Models
grand_parent: Process Models
nav_order: 3
---

# Particles - Flux Calculation
{: .fs-9 .fw-500}

---

ViennaPS integrates advanced ray tracing techniques, leveraging the power of the ViennaRay library based on Intel®'s Embree ray tracing kernel, into the realm of process simulations. This combination enables precise and efficient flux calculations within topography simulations. 
This process includes launching rays from a source and tracing their paths as they interact with the surface geometry. Ray tracing allows for the simulation of complex phenomena such as shadows, reflections, and transmission of energy through transparent materials. 

ViennaPS seamlessly integrates ViennaRay functionality through the `rayParticle` class, providing users with a versatile interface to define the key characteristics of the simulated particle species and tailor their behavior upon surface interactions, including reflective properties. Within the particle class, users can fine-tune parameters governing the initial state of particles, enabling precise control over their interactions with material surfaces. For an in-depth understanding of the `rayParticle` class and its functionalities, users are encouraged to refer to the detailed documentation available within the [ViennaRay documentation](https://viennatools.github.io/ViennaRay/particle/).

Within ViennaPS, a process model has the flexibility to encompass multiple particle species, each contributing distinct characteristics to the simulation. The fluxes computed from these particles are conveniently accessible through the [`psSurfaceModel`]({% link models/custom/surfaceModel.md %}) class. This crucial interface allows users to seamlessly integrate the particle flux data into a physical model, facilitating the simulation of intricate physical processes. By leveraging the calculated fluxes within the surface model, users can construct comprehensive simulations that capture the nuanced interplay of particles and materials, offering a robust framework for exploring diverse scenarios in process simulation.
