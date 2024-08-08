---
layout: default
title: TEOSPECVD Process
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 5
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

#  TEOS Plasma-Enhanced (PE) Chemical Vapor Deposition (CVD)
{: .fs-9 .fw-500}

```c++
#include <psTEOSPECVD.hpp>
```
---

The TEOS PE CVD process is a deposition process with an additional plasma-enhanced component, which supplies directional ions during the process. The process is specified by two particle species: the TEOS precursor radicals and the  ions. The depostion rate is controlled by a reaction order for both radicals and ions, where the final surface velocity $v$ follows:
\begin{equation}
    v = R_{rad} \cdot \Gamma_{rad}^{o_{rad}} + R_{ion} \cdot \Gamma_{ion}^{o_{ion}} 
\end{equation}
where $R_{rad}$ and $R_{ion}$ are the rates of the radicals and ions, respectively, and $\Gamma_{rad}$ and $\Gamma_{ion}$ are the fluxes of the radicals and ions, respectively. The exponents $o_{rad}$ and $o_{ion}$ are the reaction orders of the radicals and ions, respectively.

The sticking probability of the TEOS precursor radicals and the ions can be specified, as well as the exponent of the power cosine distribution of the ions. The TEOS radicals reflect diffusively from the surface, while the ions can reflect near specularly from the surface with a minimum angle specified.

## Implementation

```c++
TEOSPECVD(const NumericType pRadicalSticking, const NumericType pRadicalRate,
          const NumericType pIonRate, const NumericType pIonExponent,
          const NumericType pIonSticking = 1.,
          const NumericType pRadicalOrder = 1.,
          const NumericType pIonOrder = 1.,
          const NumericType pIonMinAngle = 0.)
```

| Parameter                  | Description                                            | Default Value          |
|----------------------------|--------------------------------------------------------|------------------------|
| `pRadicalSticking`         | Sticking probability of the TEOS precursor radicals    | 1.0                    |
| `pRadicalRate`             | Rate of the TEOS precursor radicals                    | 1.0                    |
| `pIonRate`                 | Rate of the ions                                       | 1.0                    |
| `pIonExponent`             | Exponent power cosine source distribution of the ions  | 1.0                    |
| `pIonSticking`             | Sticking probability of the ions                       | 1.0                    |
| `pRadicalOrder`            | Reaction order of the TEOS precursor radicals          | 1.0                    |
| `pIonOrder`                | Reaction order of the ions                             | 1.0                    |
| `pIonMinAngle`             | Minimum specular reflection angle of the ions          | 0.0                    |
