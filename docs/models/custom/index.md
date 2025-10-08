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

Custom process models allow users to define their own physical behavior, reaction mechanisms, or geometric transformations. A custom model can be built by combining one or more of the following core components:

* **`SurfaceModel`** — Defines surface reactions by combining incoming particle fluxes with chemical reaction kinetics.
* **`VelocityField`** — Converts the computed surface reaction rates into surface velocities used for level-set advection. If only a constant or analytic velocity is required, the `DefaultVelocityField` can be used.
* **`viennaray::Particle`** — Represents a particle species in the flux calculation, such as ions, neutrals, or byproducts. Each particle can have custom energy, angular, and sticking characteristics.
* **`AdvectionCallback`** — Optional callback function that allows executing custom operations before or after the level-set advection step (e.g., data logging, post-processing).
* **`GeometricModel`** — Used for purely geometric transformations without particle fluxes or chemistry (e.g., isotropic deposition, anisotropic etching).

{: .note }

> Custom model building is available **only in the C++ interface**.

---

## Model Composition

Analytic models (independent of particle fluxes) require only a `VelocityField`.
Flux-dependent models, on the other hand, combine:

* one or more `viennaray::Particle` instances,
* a `SurfaceModel` describing reactions, and
* a `VelocityField` to map surface rates to interface motion.

All components are assembled in a `ProcessModelCPU` (or `ProcessModelGPU`) instance, which is then passed to the `Process` class for execution.

---

## Example

```cpp
#include <psProcessModel.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>
#include <psProcess.hpp>

int main()
{
    using NumericType = double;
    constexpr int D = 3;

    // Define particles for flux calculation
    auto particle1 = std::make_unique<MyParticle1>(...);
    auto particle2 = std::make_unique<MyParticle2>(...);

    // Define surface model combining particle fluxes and reactions
    auto surfaceModel = SmartPointer<MySurfaceModel>::New(...);

    // Define velocity field (use DefaultVelocityField for simple cases)
    auto velocityField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    // Combine into a process model
    auto processModel = SmartPointer<ProcessModelCPU<NumericType, D>>::New();
    processModel->insertNextParticle(particle1);
    processModel->insertNextParticle(particle2);
    processModel->setSurfaceModel(surfaceModel);
    processModel->setVelocityField(velocityField);

    // Create and apply the process
    Process<NumericType, D> process(...);
    process.setProcessModel(processModel);
    process.apply();

    return 0;
}
```

---

This modular design enables users to flexibly construct custom models for a wide range of processes — from simple analytic etches to complex, multi-particle plasma simulations.
