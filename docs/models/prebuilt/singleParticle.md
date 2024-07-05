---
layout: default
title: Single Particle Process
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 4
---

# Single Particle Process
{: .fs-9 .fw-500}

```c++
#include <psSingleParticleProcess.hpp>
```
---

The single particle process is a simple process model that simulates either etching or deposition, assuming a single particle species. The process is specified by the rate, the particle sticking coefficient, and the exponent in the power cosine distribution of the initial particle directions. 

The rate can be either negative or positive, corresponding to etching or deposition, respectively. The sticking coefficient is the probability that a particle will stick to the surface upon impact, thus controlling the number of diffusive reflections from the surface. The exponent in the power cosine distribution of the initial particle directions is a measure of the angular spread of the initial particle directions. A higher exponent corresponds to a more focused beam of particles.

Additionally, mask materials can be specified, where the rate is assumed to be zero. 

## Implementation

```c++
SingleParticleProcess(const NumericType rate = 1.,
                      const NumericType stickingProbability = 1.,
                      const NumericType sourceDistributionPower = 1.,
                      const Material maskMaterial = Material::None)

SingleParticleProcess(const NumericType rate,
                      const NumericType stickingProbability,
                      const NumericType sourceDistributionPower,
                      const std::vector<Material> maskMaterials) 

```

| Parameter                  | Description                                            | Default Value          |
|----------------------------|--------------------------------------------------------|------------------------|
| `rate`                      | Rate of the single particle process                    | 1.0                    |
| `stickingProbability`       | Sticking probability of particles                      | 1.0                    |
| `sourceDistributionPower`   | Power of the power cosine source distribution          | 1.0                    |
| `maskMaterial`              | Mask material                       | `Material::None`       |

__Example usage__:

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
```c++
...
// for a single mask material
auto model = SmartPointer<SingleParticleProcess<NumericType, D>>::New(1., 0.1, 1., maskMaterial);
...
// for multiple mask materials
auto model = SmartPointer<SingleParticleProcess<NumericType, D>>::New(1., 0.1, 1., {mask1, mask2});
...
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green}
</summary>
```python
...
model = vps.SingleParticleProcess(rate=1., stickingProbability=0.1, sourceExponent=1., maskMaterials=[maskMaterial])
...
```
</details>

## Related Examples

* [Trench Deposition](https://github.com/ViennaTools/ViennaPS/tree/master/examples/trenchDeposition)
