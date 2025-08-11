---
layout: default
title: Ion Beam Etching
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 9
---

# Ion Beam Etching (IBE) Process
{: .fs-9 .fw-500}

```c++
#include <psIonBeamEtching.hpp>
```
---

The **IonBeamEtching** model simulates anisotropic material removal by a directed ion beam, optionally including **redeposition** of sputtered material.
It uses a physics-based yield function to compute the local etch rate as a function of incidence angle and ion energy, making it suitable for realistic simulation of focused or broad-beam ion etching setups.

The process can be configured with:

* **Mask materials** – specified as a list of `Material` types that are excluded from etching.
* **IBE parameters** – a complete set of physical and process-specific properties, such as ion energy distribution, sputter yield function, incidence angles, redeposition thresholds, and material-specific etch rates.

During simulation:

* Ion trajectories are traced using ViennaPS’ ray-tracing engine.
* The **surface model** calculates local etch velocities from the simulated ion flux, redeposition flux, and yield function.
* **Particle–surface interactions** determine whether an ion sputters material, is reflected, or causes redeposition, with reflection modeled via a coned-cosine distribution.

By adjusting the **`IBEParameters`** and mask configuration, the model can replicate a wide range of IBE scenarios—from purely directional sputtering to processes with significant redeposition effects.

---

| Parameter               | Type       | Description                                                                                          | Units / Range                   | Default |
|-------------------------|------------|------------------------------------------------------------------------------------------------------|----------------------------------|---------|
| `planeWaferRate`        | double     | Base etch rate for a reference (plane) wafer surface.                                                | User-defined (e.g., nm/min)      | 1.0     |
| `materialPlaneWaferRate`| map        | Material-specific plane wafer rates overriding `planeWaferRate`.                                     | User-defined per `Material`      | —       |
| `meanEnergy`            | double     | Mean ion energy in the beam.                                                                         | eV                               | 250     |
| `sigmaEnergy`           | double     | Standard deviation of the ion energy distribution.                                                   | eV                               | 10      |
| `thresholdEnergy`       | double     | Minimum ion energy required for sputtering.                                                          | eV                               | 20      |
| `exponent`              | double     | Exponent controlling the angular distribution of the ion source.                                     | > 1                              | 100     |
| `n_l`                   | double     | Shape parameter for the reflection energy distribution.                                              | > 1                              | 10      |
| `inflectAngle`          | double     | Inflection angle for energy reflection behavior.                                                      | degrees                          | 89      |
| `minAngle`              | double     | Minimum angle for coned reflection.                                                                  | degrees                          | 85      |
| `tiltAngle`             | double     | Tilt angle of the incoming ion beam relative to surface normal.                                      | degrees                          | 0       |
| `yieldFunction`         | function   | User-defined sputter yield as a function of incidence angle `θ` (in radians).                        | —                                | `1.0`   |
| `redepositionThreshold` | double     | Minimum sputtered particle energy or yield before redeposition is considered.                        | User-defined                     | 0.1     |
| `redepositionRate`      | double     | Fraction of sputtered material redeposited on the surface.                                           | 0.0 – 1.0                        | 0.0     |


## Related Examples

* [Blazed Gratings Etching](https://github.com/ViennaTools/ViennaPS/tree/master/examples/blazedGratingsEtching)
