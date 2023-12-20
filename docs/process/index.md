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
The `psProcess` class functions as the primary simulation interface, consolidating crucial elements such as the simulation domain, process model, process duration, and requisite ray-tracing parameters. This interface also encompasses the necessary methods for configuring these attributes. Upon setting these parameters, the `apply()` method is employed to execute the process, initiating and conducting the simulation.

__Example usage:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
```c++
...
psProcess<NumericType, D> process;
process.setDomain(myDomain);
process.setProcessModel(myModel);
process.setProcessDuration(10.);
process.setNumberOfRaysPerPoint(1000);
process.enableFluxSmoothing();
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
process.setNumberOfRaysPerPoint(1000)
process.enableFluxSmoothing()
process.apply()
...
```
</details>

## Member Functions
### Constructors
```c++
// Default constructor
psProcess()
// Constructor from domain
psProcess(psSmartPointer<psDomain<NumericType, D>> passedDomain)
// Constructor from domain, process model, and duration, 
// to apply simple processes
template <typename ProcessModelType>
psProcess(psSmartPointer<psDomain<NumericType, D>> passedDomain,
        psSmartPointer<ProcessModelType> passedProcessModel,
        const NumericType passedDuration = 0.)
```
In summary, these constructors provide different ways to create a `psProcess` object, allowing for flexibility depending on what data is available at the time of object creation.
1. The first constructor is a default constructor. It's defined as `psProcess()` and it doesn't take any arguments. This constructor allows for the creation of a `psProcess` object without any initial values.
2. The second constructor takes a single argument: a smart pointer to a `psDomain` object. This constructor initializes the domain member variable of the `psProcess` class with the passed `psDomain` object.
3. The third constructor is a template constructor that takes three arguments: a smart pointer to a `psDomain` object, a smart pointer to a `ProcessModelType` object, and a `NumericType` representing the process duration. This constructor initializes the domain and processDuration member variables with the passed values and also sets the model member variable to the dynamically cast ProcessModelType object. This allows the user to run a process from an anonymous object. For example:
```cpp
psProcess<NumericType, D>(myDomain, myModel, processDuration).apply()
```

---
### Set the domain
```cpp
void setDomain(psSmartPointer<psDomain<NumericType, D>> passedDomain)
```
Sets the process domain. 

---
### Set the process model
```c++
template <typename ProcessModelType>
void setProcessModel(psSmartPointer<ProcessModelType> passedProcessModel)
```
Sets the process model. This can be either a pre-configured process model or a custom process model. A custom process model must interface the `psProcessModel` class.

---
### Set the source direction
```c++
void setSourceDirection(const rayTraceDirection passedDirection)
```
Set the source direction, where the rays should be traced from. The passed direction parameter is using the enum `rayTraceDirection` which contains the following values:
`POS_X`, `NEG_X`, `POS_Y`, `NEG_Y`, `POS_Z`, `NEG_Z`.

---
### Set the process duration
```c++
void setProcessDuration(NumericType passedDuration)
```
Specifies the duration of the process. If the process duration is set to 0, exclusively the advection callback `applyPreAdvect()` is executed on the domain. This feature is particularly useful for applying only a volume model without engaging in further simulation steps.

---
### Set the number of rays to be traced
```c++
void setNumberOfRaysPerPoint(unsigned raysPerPoint)
```
Specify the number of rays to be traced for each particle throughout the process. The total count of rays is the product of this number and the number of points in the process geometry.

---
### Set the number of coverage initialization iterations
```c++
void setMaxCoverageInitIterations(unsigned numIter)
```
Set the number of iterations to initialize the coverages.

---
### Enable or disable flux smoothing
```c++
void enableFluxSmoothing()
void disableFluxSmoothing()
```
Toggle the option to enable or disable flux smoothing. When flux smoothing is activated, the flux at each surface point, computed by the ray tracer, undergoes averaging across the neighboring surface points. Per default, flux smoothing is enabled.

---
### Set the integration scheme
```c++
void
setIntegrationScheme(lsIntegrationSchemeEnum passedIntegrationScheme)
```
Set the integration scheme for solving the level-set equation.
Possible integration schemes are specified in lsIntegrationSchemeEnum.

---
### Set the time step ratio
```c++
void setTimeStepRatio(NumericType cfl)
```
Set the CFL (Courant-Friedrichs-Levy) condition to use during surface advection in the level-set. The CFL condition defines the maximum distance a surface is allowed to move in a single advection step. It MUST be below 0.5 to guarantee numerical stability. Defaults to 0.4999.

---
### Set time interval to save intermediate results
```c++
void setPrintTimeInterval(NumericType passedTime)
```
Sets the minimum time between printing intermediate results during the process. If this is set to a non-positive value, no intermediate results are printed.