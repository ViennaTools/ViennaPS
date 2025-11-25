---
layout: default
title: TEOS Deposition
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 6
---
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  }
};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>

#  TEOS Deposition
{: .fs-9 .fw-500}

```c++
#include <psTEOSDeposition.hpp>
```
CPU only
{: .label .label-yellow}
---

The **TEOSDeposition** model simulates conformal deposition of silicon dioxide films from tetraethyl orthosilicate (TEOS) precursors.
In this implementation, surface growth rates are calculated directly from the local particle flux, following a power-law dependence defined by a user-specified deposition rate and reaction order.

Two configurations are supported:

* **Single-particle model** – Represents a single precursor species interacting with the surface. The surface velocity is determined from one particle flux channel, and coverages can be updated dynamically based on the incoming flux.
* **Multi-particle model** – Supports two distinct precursor or reactant species, each with independent sticking probability, deposition rate, and reaction order. The total growth rate is computed as the sum of the contributions from both species.

Particle–surface interactions are handled via diffuse reflection, with the sticking probability modulated by the local surface coverage (in the single-particle case). 
By adjusting the sticking probability, deposition rate, and reaction order parameters, this process can be tuned to match experimental data for a wide range of TEOS-based plasma-enhanced or thermal CVD conditions.

## Implementation

```c++
TEOSDeposition(const NumericType stickingProbabilityP1, const NumericType rateP1,
               const NumericType orderP1, const NumericType stickingProbabilityP2 = 0.,
               const NumericType rateP2 = 0.,
               const NumericType orderP2 = 0.)
```

| Parameter        | Type    | Description                                                                 | Units / Range                 | Mode              |
|------------------|---------|-----------------------------------------------------------------------------|--------------------------------|-------------------|
| `stickingProbabilityP1`    | double  | Sticking probability of first particle type                                 | 0.0 – 1.0                      | Single & Multi    |
| `rateP1`        | double  | Deposition rate scaling factor for first particle type                      | User-defined (e.g., nm/min)    | Single & Multi    |
| `orderP1`       | double  | Reaction order for first particle type                                      | > 0                            | Single & Multi    |
| `stickingProbabilityP2`    | double  | Sticking probability of second particle type                                | 0.0 – 1.0                      | Multi only        |
| `rateP2`        | double  | Deposition rate scaling factor for second particle type                     | User-defined (e.g., nm/min)    | Multi only        |
| `orderP2`       | double  | Reaction order for second particle type                                     | > 0                            | Multi only        |
