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

## Summary

- The `psSurfaceModel` class is used to describe surface reactions, combining particle fluxes with the surface chemical reactions.
- The velocities used for surface advection in a time step are calculated through the `calculateVelocities()` function.
- Surface coverages can be used to track the coverage a chemical species on the surface through a time step.
- The coverages can be initialized to equilibrium by iteratively calculating the fluxes on the surface and updating the coverages. The number of iterations to initialize the coverages can be specified through the [`psProcess`]({% link process/index.md %}).
- Coverages and fluxes are stored as `psPointData`.

---

The `psSurfaceModel` class serves as a comprehensive framework for detailing surface chemistries. Users have the flexibility to create a customized child class where they can precisely dictate how surface coverages evolve, driven by the rates at which particles impact the surface. 

One key feature is the capability to monitor surface coverages, providing insights into the presence of chemical species on the surface throughout a simulation's time step. 
To initialize the coverage data vector, the method `initializeCoverages()` is employed. 
```c++
  void initializeCoverages(unsigned numSurfacePoints) override {
    // a single set of coverages is initialized here
    std::vector<NumericType> someCoverage(numSurfacePoints, 0);

    coverages = psSmartPointer<psPointData<NumericType>>::New();
    coverages->insertNextScalarData(someCoverage, "someCoverage");
  }
```
To use coverages, it is essential to initialize the class member `coverages` with a new instance of `psPointData`. If the `coverages` variable is left as `nullptr`, no coverages will be utilized during the simulation. To initialize a single coverage, a container with a size equal to the number of surface points must be created and inserted into the `psPointData`. Additionally, a name for the coverage can be specified during initialization. This designated name should then be used in `updateCoverages()` or `calculateVelocities()` to access the specific coverage as needed.

To ensure accurate representations, coverages can be initialized to equilibrium by iteratively calculating surface fluxes and updating coverages. The initialization process's iteration count is customizable through the [psProcess]({% link process/index.md %}) interface. 
The method `updateCoverages()` encapsulates the user-defined description of surface coverage evolution in each iteration. Since `coverages` is a member of the `psSurfaceModel` class, it can be accessed in every member function.
```c++
  void updateCoverages(psSmartPointer<psPointData<NumericType>> particleFluxes,
                       const std::vector<NumericType> &materialIds) override {
    auto myCoverage = coverages->getScalarData("someCoverage");
    // update coverage from calculated fluxes
  }
```

Within the `psSurfaceModel` class, the method `calculateVelocities()` utilizes fluxes obtained through ray tracing, to provide the velocities used for surface advection in a time step. Here the fluxes from particle, as well as previously calculated coverages can be accessed and combined to yield the final velocity at each surface point. The function should return a `psSmartPointer` to a new vector, containing the velocity at each surface point.

In order to create a custom surface the user has to interface the `psSurfaceModel` class. An example implementation of a custom surface model is given below:
```c++
template <typename NumericType>
class myCustomSurfaceModel : public psSurfaceModel<NumericType> {
public:
  using psSurfaceModel<NumericType>::coverages; // needed to access coverages

  void initializeCoverages(unsigned numSurfacePoints) override {
    // a single set of coverages is initialized here
    std::vector<NumericType> someCoverage(numSurfacePoints, 0);

    coverages = psSmartPointer<psPointData<NumericType>>::New();
    coverages->insertNextScalarData(someCoverage, "someCoverage");
  }

  void updateCoverages(psSmartPointer<psPointData<NumericType>> particleFluxes,
                       const std::vector<NumericType> &materialIds) override {
    auto myCoverage = coverages->getScalarData("someCoverage");
    // update coverage from calculated fluxes
  }

  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {
    // use coverages and rates here to calculate the velocity here
    return psSmartPointer<std::vector<NumericType>>::New(
        *rates->getScalarData("particleRate"));
  }
};
```