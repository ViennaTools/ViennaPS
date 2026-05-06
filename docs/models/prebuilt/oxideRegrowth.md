---
layout: default
title: Oxide Regrowth
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 13
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

# Oxide Regrowth
{: .fs-9 .fw-500}

```c++
#include <psOxideRegrowth.hpp>
```
---

`OxideRegrowth` models oxide regrowth during selective etching of alternating SiO<sub>2</sub> and Si<sub>3</sub>N<sub>4</sub> stacks. The model combines selective etching, byproduct dynamics, diffusion, and redeposition through a velocity field and advection callback.

It is based on the model presented in [Modeling Oxide Regrowth During Selective Etching in Vertical 3D NAND Structures](https://doi.org/10.23919/SISPAD57422.2023.10319506).

## Constructor

```c++
OxideRegrowth(NumericType nitrideEtchRate,
              NumericType oxideEtchRate,
              NumericType redepositionRate,
              NumericType reDepositionThreshold,
              NumericType redepositionTimeInt,
              NumericType diffusionCoefficient,
              NumericType sinkStrength,
              NumericType scallopVelocity,
              NumericType centerVelocity,
              NumericType topHeight,
              NumericType centerWidth,
              NumericType timeStabilityFactor = 0.245)
```

| Parameter | Description |
|-----------|-------------|
| `nitrideEtchRate` | Etch rate for nitride regions. |
| `oxideEtchRate` | Etch rate for oxide regions. |
| `redepositionRate` | Rate used for oxide redeposition. |
| `reDepositionThreshold` | Byproduct threshold above which redeposition is enabled. |
| `redepositionTimeInt` | Time interval used by the redeposition model. |
| `diffusionCoefficient` | Diffusion coefficient for byproduct transport. |
| `sinkStrength` | Strength of the byproduct sink term. |
| `scallopVelocity` | Velocity contribution controlling scallop evolution. |
| `centerVelocity` | Velocity contribution around the stack center. |
| `topHeight` | Height of the top reference region. |
| `centerWidth` | Width of the center region. Internally, half of this value is used. |
| `timeStabilityFactor` | Stability factor for the callback time stepping. |

## Example Usage

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
```c++
auto model = SmartPointer<OxideRegrowth<NumericType, D>>::New(
    nitrideEtchRate, oxideEtchRate, redepositionRate,
    reDepositionThreshold, redepositionTimeInt, diffusionCoefficient,
    sinkStrength, scallopVelocity, centerVelocity, topHeight, centerWidth);
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green}
</summary>
```python
model = vps.OxideRegrowth(
    nitrideEtchRate, oxideEtchRate, redepositionRate,
    reDepositionThreshold, redepositionTimeInt, diffusionCoefficient,
    sinkStrength, scallopVelocity, centerVelocity, topHeight, centerWidth
)
```
</details>

## Related Examples

* [Oxide Regrowth](https://github.com/ViennaTools/ViennaPS/tree/master/examples/oxideRegrowth)
