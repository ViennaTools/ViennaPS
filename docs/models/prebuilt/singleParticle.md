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

Additionally, mask materials can be specified, where the rate is assumed to be zero, or map between material and rate can be defined. 

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

SingleParticleProcess(std::unordered_map<Material, NumericType> materialRates,
                      NumericType stickingProbability,
                      NumericType sourceDistributionPower)

```

| Parameter                  | Description                                            | Default Value          |
|----------------------------|--------------------------------------------------------|------------------------|
| `rate`                     | Default rate of the single particle process, if no material specific rates are defined   | 1.0              |
| `stickingProbability`      | Sticking probability of particles                      | 1.0                    |
| `sourceDistributionPower`  | Power of the power cosine source distribution          | 1.0                    |
| `maskMaterial`             | Mask material                       | `Material::None`       |

Rates can also be specified for specific materials using a map between material and rate.

__Example usage__:

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
{% raw %}
```c++
...
// for a single mask material
auto model = SmartPointer<SingleParticleProcess<NumericType, D>>::New(1., 0.1, 1., maskMaterial);
...
// for multiple mask materials
auto model = SmartPointer<SingleParticleProcess<NumericType, D>>::New(1., 0.1, 1., {mask1, mask2});

// for material specific rates
std::unordered_map<Material, NumericType> materialRates = {{Material::Si, 1.}, {Material::SiO2, 0.5}};
auto model = SmartPointer<SingleParticleProcess<NumericType, D>>::New(materialRates, 0.1, 1.);
...
```
{% endraw %}
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green}
</summary>
{% raw %}
```python
...
model = vps.SingleParticleProcess(rate=1., stickingProbability=0.1, sourceExponent=1., maskMaterials=[maskMaterial])

# using material specific rates
rates = {vps.Material.Si: 1., vps.Material.SiO2: 0.5}
model = vps.SingleParticleProcess(materialRates=rates, stickingProbability=0.1, sourceExponent=1.)
...
```
{% endraw %}
</details>

## Related Examples

* [Trench Deposition](https://github.com/ViennaTools/ViennaPS/tree/master/examples/trenchDeposition)
