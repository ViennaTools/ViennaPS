---
layout: default
title: TEOS PE CVD Process
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 7
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
TEOSPECVD(const NumericType radicalSticking, const NumericType radicalRate,
          const NumericType ionRate, const NumericType ionExponent,
          const NumericType ionSticking = 1.,
          const NumericType radicalOrder = 1.,
          const NumericType ionOrder = 1.,
          const NumericType ionMinAngle = 0.)
```

| Parameter                  | Description                                            | Default Value          |
|----------------------------|--------------------------------------------------------|------------------------|
| `radicalSticking`         | Sticking probability of the TEOS precursor radicals    | 1.0                    |
| `radicalRate`             | Rate of the TEOS precursor radicals                    | 1.0                    |
| `ionRate`                 | Rate of the ions                                       | 1.0                    |
| `ionExponent`             | Exponent power cosine source distribution of the ions  | 1.0                    |
| `ionSticking`             | Sticking probability of the ions                       | 1.0                    |
| `radicalOrder`            | Reaction order of the TEOS precursor radicals          | 1.0                    |
| `ionOrder`                | Reaction order of the ions                             | 1.0                    |
| `ionMinAngle`             | Minimum specular reflection angle of the ions          | 0.0                    |
