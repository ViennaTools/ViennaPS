---
layout: default
title: Surface Model
parent: Custom Models
grand_parent: Process Models
nav_order: 1
---

# Surface Model
{: .fs-9 .fw-500}

---

The `psSurfaceModel` class serves as a comprehensive framework for detailing surface chemistries. Users have the flexibility to create a customized child class where they can precisely dictate how surface coverages evolve, driven by the rates at which particles impact the surface. The method `updateCoverages()` encapsulates the user-defined description of surface coverage evolution.

Furthermore, within `psSurfaceModel`, the method `calculateVelocities()` utilizes data obtained through ray tracing, providing velocities crucial for surface advection. To initialize the coverage data vector, the method `initializeCoverages()` is employed, assuming an equilibrium state. Users can define the container size with values initialized to $0$ at each surface point, initiating ray tracing until equilibrium is attained. While our default implementation typically converges in around $10$ iterations, users can easily customize this parameter based on their specific simulation requirements.

In order to create a custom surface the user has to interface the `psSurfaceModel` class. An example implementation of a custom surface model is given below:
```c++
template <typename NumericType>
class myCustomSurfaceModel : public psSurfaceModel<NumericType> {
public:
  using psSurfaceModel<NumericType>::coverages;
  using psSurfaceModel<NumericType>::processParams;

  void initializeCoverages(unsigned numGeometryPoints) override {
    // a single set of coverages in initialized here
    std::vector<NumericType> someCoverages(numGeometryPoints, 0);

    coverages = psSmartPointer<psPointData<NumericType>>::New();
    coverages->insertNextScalarData(someCoverages, "coverages");
  }

  void initializeProcessParameters() override {
    // a single process parameter is initialized here
    processParams = psSmartPointer<psProcessParams<NumericType>>::New();
    processParams->insertNextScalar(0., "processParameter");
  }

  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {
    // use coverages and rates here to calculate the velocity here
    return psSmartPointer<std::vector<NumericType>>::New(
        *rates->getScalarData("particleRate"));
  }

  void updateCoverages(psSmartPointer<psPointData<NumericType>> rates,
                       const std::vector<NumericType> &materialIds) override {
    // update coverages here
  }
};
```