---
layout: default
title: Dependencies
nav_order: 2
---

# Dependencies
{: .fs-9 .fw-700 }

---

## [ViennaLS](https://github.com/ViennaTools/ViennaLS)

ViennaLS forms the foundation of the process simulator, applying the level-set surface representation concepts for topography simulations. This module not only stores the level-set surface but also encompasses essential algorithms for geometry initialization, level-set value manipulation based on a velocity field, surface feature analysis, and seamless conversion of the level-set representation to other commonly employed material representations in device simulators.

## [ViennaRay](https://github.com/ViennaTools/ViennaRay)

ViennaPS relies on ViennaRay, a top-down Monte Carlo flux calculation library, to carry out essential flux calculations. This library is built upon Intel®'s ray tracing kernel, [Embree](https://www.embree.org/).
Crafted with a focus on efficiency and high-performance ray tracing, ViennaRay ensures a seamless user experience through its straightforward and user-friendly interface.

In the top-down Monte Carlo approach, a large number of pseudo-particles are launched from a source plane situated above the surface, and their points of impact on the surface are determined.
These pseudo-particles are initialized with a uniform initial position on the source plane and an initial direction that follows a power-cosine distribution. Each pseudo-particle carries a specific payload, representing a fraction of the total source flux. Upon reaching the surface, the current payload of the pseudo-particle contributes to the flux at that particular surface location.

Furthermore, pseudo-particles have the capability to undergo reflection from the surface. The payload of a pseudo-particle undergoes reduction by the sticking coefficient during reflection. As a result, a pseudo-particle is tracked until its payload falls below a certain threshold or until it exits the simulation domain. This tracking mechanism provides a comprehensive understanding of the particle dynamics during its interaction with the sample surface.
