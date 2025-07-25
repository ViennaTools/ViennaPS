---
layout: default
title: Running a Process
nav_order: 8
---

# Running a Process
{: .fs-9 .fw-700}
```c++
#include <psProcess.hpp>
```
---

![]({% link assets/images/process.png %})

---

The `Process` class functions as the primary simulation interface, consolidating crucial elements such as the simulation domain, process model, process duration, and requisite ray-tracing parameters. This interface also contains the necessary methods for configuring these attributes. Upon setting these parameters, the `apply()` method is employed to execute the process,.

__Example usage:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
```c++
// namespace viennaps
...
Process<NumericType, D> process;
process.setDomain(myDomain);
process.setProcessModel(myModel);
process.setProcessDuration(10.);
process.apply();
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
process = vps.Process()
process.setDomain(myDomain)
process.setProcessModel(myModel)
process.setProcessDuration(10.)
process.apply()
...
```
</details>

## Process Parameters

{: .note }
> Advanced process parameters, as described in this section are available from version 3.3.0

Expert users can set specific parameter for the Level-Set integration and Ray Tracing flux calculation steps using the advanced parameters structs `AdvectionParameters` and `RayTracingParameters`. These parameter structs contain:

__AdvectionParameters:__

| Parameter            | Type         | Default Value                                      | Description |
|----------------------|-------------|----------------------------------------------------|-------------|
| `integrationScheme`  | `IntegrationSchemeEnum` | `ENGQUIST_OSHER_1ST_ORDER` | Integration scheme used for advection. For options see [here](https://viennatools.github.io/ViennaLS/namespaceviennals.html#a939e6f11eed9a003a0723a255290377f). |
| `timeStepRatio`      | `NumericType` | `0.4999` | Ratio controlling the time step specified by the CFL condition. More [details](https://viennatools.github.io/ViennaLS/classviennals_1_1Advect.html#ad6aba52d0b3c5fb9fcb4e83da138573c) |
| `dissipationAlpha`   | `NumericType` | `1.0` | Factor controlling dissipation in Lax-Friedrichs type integration schemes. |
| `velocityOutput`     | `bool` | `false` | Whether to output velocity data for each advection step. |
| `ignoreVoids`        | `bool` | `false` | Whether to ignore void regions. |

__RayTracingParameters:__

| Parameter            | Type                        | Default Value                             | Description |
|----------------------|---------------------------|-------------------------------------------|-------------|
| `sourceDirection`    | `TraceDirection`          | `POS_Z` (if `D == 3`), `POS_Y` (otherwise) | Direction of the ray source. |
| `normalizationType`  | `NormalizationType`       | `SOURCE`                                  | Type of normalization used. Other option `MAX`. |
| `raysPerPoint`       | `unsigned`                | `1000`                                    | Number of rays to trace per point in the geometry. |
| `diskRadius`        | `NumericType`             | `0`                                      | Radius of the disks in the ray tracing geometry. If this value is 0 the default disk radius is used, which is the minimum radius such that there are no holes in the geometry. |
| `useRandomSeeds`     | `bool`                     | `true`                                    | Whether to use random seeds. |
| `ignoreFluxBoundaries` | `bool`                 | `false`                                   | Whether to ignore boundary condtions during ray tracing. |
| `smoothingNeighbors` | `int`                     | `1`                                       | Number of neighboring points used for smoothing the flux after ray tracing. |

__Example usage:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
```c++
// namespace viennaps
...
AdvectionParameters<NumericType> advParams;
advParams.integrationScheme = viennals::IntegrationSchemeEnum::LOCAL_LAX_FRIEDRICHS_2ND_ORDER
advParams.timeStepRatio = 0.25
advParams.dissipationAlpha = 2.0

RayTracingParameters<NumericType, D> tracingParams;
tracingParams.raysPerPoint = 500
tracingParams.smoothingNeighbors = 0 // disable flux smoothing

Process<NumericType, D> process(myDomain, myModel, duration);
process.setAdvectionParameters(advParams)
process.setRayTracingParameters(tracingParams)
process.apply();
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
advParams = vps.AdvectionParameters()
advParams.integrationScheme = vps.ls.IntegrationSchemeEnum.LOCAL_LAX_FRIEDRICHS_2ND_ORDER
advParams.timeStepRatio = 0.25
advParams.dissipationAlpha = 2.0

tracingParams = vps.RayTracingParameters()
tracingParams.raysPerPoint = 500
tracingParams.smoothingNeighbors = 0 # disable flux smoothing

process = vps.Process(myDomain, myModel, duration)
process.setAdvectionParameters(advParams)
process.setRayTracingParameters(tracingParams)
process.apply()
...
```
</details>

## Member Functions
### Constructors
```c++
// Default constructor
Process()
// Constructor from domain
Process(SmartPointer<Domain<NumericType, D>> passedDomain)
// Constructor from domain, process model, and duration, 
// to apply simple processes
template <typename ProcessModelType>
Process(SmartPointer<Domain<NumericType, D>> passedDomain,
        SmartPointer<ProcessModelType> passedProcessModel,
        const NumericType passedDuration = 0.)
```
In summary, these constructors provide different ways to create a `Process` object, allowing for flexibility depending on what data is available at the time of object creation.
1. The first constructor is a default constructor. It's defined as `Process()` and it doesn't take any arguments. This constructor allows for the creation of a `Process` object without any initial values.
2. The second constructor takes a single argument: a smart pointer to a `Domain` object. This constructor initializes the domain member variable of the `Process` class with the passed `Domain` object.
3. The third constructor is a template constructor that takes three arguments: a smart pointer to a `Domain` object, a smart pointer to a `ProcessModelType` object, and a `NumericType` representing the process duration. This constructor initializes the domain and processDuration member variables with the passed values and also sets the model member variable to the dynamically cast ProcessModelType object. This allows the user to run a process from an anonymous object. For example:
```cpp
Process<NumericType, D>(myDomain, myModel, processDuration).apply()
```

---
### Set the domain
```cpp
void setDomain(SmartPointer<Domain<NumericType, D>> passedDomain)
```
Sets the process domain. 

---
### Set the process model
```c++
void setProcessModel(SmartPointer<ProcessModel<NumericType, D>> passedProcessModel)
```
Sets the process model. This can be either a pre-configured process model or a custom process model. 

---
### Set the source direction
```c++
void setSourceDirection(const viennaray::TraceDirection passedDirection)
```
Set the source direction, where the rays should be traced from. The passed direction parameter is using the enum `viennaray::TraceDirection` which contains the following values:
`POS_X`, `NEG_X`, `POS_Y`, `NEG_Y`, `POS_Z`, `NEG_Z`.

---
### Set the process duration
```c++
void setProcessDuration(NumericType passedDuration)
```
Specifies the duration of the process. If the process duration is set to 0, exclusively the advection callback `applyPreAdvect()` is executed on the domain. This feature is particularly useful for applying only a volume model without engaging in further simulation steps.

---
### Single-Pass Flux Calculation
```c++
SmartPointer<viennals::Mesh<NumericType>> calculateFlux()
```
Calculate the flux(es) for the current process. This function returns a smart pointer to a `viennals::Mesh<NumericType>` object containing the disk mesh and flux data.

---
### Set the number of coverage initialization iterations
```c++
void setMaxCoverageInitIterations(unsigned numIter)
```
Set the maximum number of iterations to initialize the coverages. If additionally the coverage delta threshold is set, the coverage initialization is considered converged if the coverage delta is below the threshold or the maximum number of iterations is reached.

--- 
### Set coverage delta threshold
```c++
void setCoverageDeltaThreshold(NumericType threshold)
```
Set the threshold for the coverage delta metric to reach convergence. If the coverage delta is below this threshold, the coverage initialization is considered converged. During a process, if the coverage delta is higher than this threshold, the flux calculation is repeated.

---
### Set the advection parameters
```c++
void setAdvectionParameters(const AdvectionParameters<NumericType>& passedAdvectionParameters)
```
Set the advection parameters for the level-set integration. The advection parameters are defined in the `AdvectionParameters` struct.

---
### Set the ray tracing parameters
```c++
void setRayTracingParameters(const RayTracingParameters<NumericType, D>& passedRayTracingParameters)
```
Set the ray tracing parameters for the flux calculation. The ray tracing parameters are defined in the `RayTracingParameters` struct.

---

{: .note }
> The following functions are used to set parameters in the `AdvectionParameters` and `RayTracingParameters` structs. 
> From version **3.3.0**, it is recommended to use the `setAdvectionParameters` and `setRayTracingParameters` functions to set these parameters.

---
### Set the number of rays to be traced
```c++
void setNumberOfRaysPerPoint(unsigned raysPerPoint)
```
Specify the number of rays to be traced for each particle throughout the process. The total count of rays is the product of this number and the number of points in the process geometry.

---
### Enable or disable flux smoothing
```c++
void enableFluxSmoothing()
void disableFluxSmoothing()
```
Toggle the option to enable or disable flux smoothing. When flux smoothing is activated, the flux at each surface point, computed by the ray tracer, undergoes averaging across the neighboring surface points. Per default, flux smoothing is enabled.

---
### Enable or disable flux boundary conditions
```c++
void enableFluxBoundaries()
void disableFluxBoundaries()
```
Toggle the option to enable or disable flux boundary conditions. When flux boundary conditions are disabled, particles are allowed to leave the domain without being reflected back into the domain. Per default, flux boundary conditions are enabled.

---
### Enable or disable random seeds
```c++
void enableRandomSeeds()
void disableRandomSeeds()
```
Toggle the option to enable or disable random seeds. When random seeds are enabled, the random number generator is seeded with a random value, making the results generally not perfectly reproducible. Per default, random seeds are enabled.

---
### Set the integration scheme
```c++
void
setIntegrationScheme(lsIntegrationSchemeEnum passedIntegrationScheme)
```
Set the integration scheme for solving the level-set equation.
Possible integration schemes are specified in [`viennals::IntegrationSchemeEnum`](https://viennatools.github.io/ViennaLS/namespaceviennals.html#a939e6f11eed9a003a0723a255290377f).

---
### Set the time step ratio
```c++
void setTimeStepRatio(NumericType cfl)
```
Set the CFL (Courant-Friedrichs-Levy) condition to use during surface advection in the level-set. The CFL condition defines the maximum distance a surface is allowed to move in a single advection step. It MUST be below 0.5 to guarantee numerical stability. Defaults to 0.4999.

