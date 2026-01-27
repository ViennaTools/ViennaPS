---
layout: default
title: Running a Process
nav_order: 8
has_children: true
---

# Running a Process
{: .fs-9 .fw-700}

```c++
#include <psProcess.hpp>
```

![]({% link assets/images/process.png %})

---

The `Process` class is the main simulation interface. It holds the domain, the process model, the duration, and advanced parameters. Configure it, then call `apply()`. The passed domain is modified in place.

<!-- ## What’s new in 4.0.0

* **Flux engine switch** on the process via `setFluxEngineType(...)`.

  * Options: `AUTO` *(default)*, `CPU_DISK`, `GPU_DISK`, `GPU_LINE`, `GPU_TRIANGLE`.
  * `AUTO` selects CPU or GPU based on build and model availability.
* **Unified parameter API**: set all parameter structs via `setParameters(...)`.

  * Supported structs: `AdvectionParameters`, `RayTracingParameters`, `CoverageParameters`, `AtomicLayerProcessParameters`.
* **AtomicLayerProcess removed**. The standard `Process()` detects ALP behavior from the selected model.
* **Python bindings unified**. Use `viennaps` (with `viennaps.d2` / `viennaps.d3`). Change default dimension via `viennaps.setDimension()`. -->

---

## Example usage

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>

```c++
// namespace viennaps
using T = double;
constexpr int D = 3;

auto process = ps::Process<T, D>();
process.setDomain(myDomain);
process.setProcessModel(myModel);
process.setProcessDuration(10.0);

// Flux engine selection
process.setFluxEngineType(ps::FluxEngineType::AUTO);

// Optional parameters
ps::AdvectionParameters<T> adv;
adv.timeStepRatio = 0.25;

ps::RayTracingParameters<T, D> rt;
rt.raysPerPoint = 500;

ps::CoverageParameters<T> cov;
cov.maxIterations = 10;

ps::AtomicLayerProcessParameters<T> alp;
alp.numCycles = 2;

process.setParameters(adv);
process.setParameters(rt);
process.setParameters(cov);
process.setParameters(alp);

// Run
process.apply();
```

</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green}
</summary>

```python
import viennaps as vps

process = vps.Process()
process.setDomain(myDomain)
process.setProcessModel(myModel)
process.setProcessDuration(10.0)

# Flux engine selection
process.setFluxEngineType(vps.FluxEngineType.AUTO)

# Optional parameters
adv = vps.AdvectionParameters()
adv.timeStepRatio = 0.25

rt = vps.RayTracingParameters()
rt.raysPerPoint = 500

cov = vps.CoverageParameters()
cov.maxIterations = 10

alp = vps.AtomicLayerProcessParameters()
alp.numCycles = 2

process.setParameters(adv)
process.setParameters(rt)
process.setParameters(cov)
process.setParameters(alp)

process.apply()
```

</details>

---

## Process parameters 

All advanced parameters are set via `setParameters(...)`.

---

## Flux engine

Select the flux computation method at runtime.

```c++
// C++
process.setFluxEngineType(ps::FluxEngineType::AUTO);       // default
// or: CPU_DISK, GPU_DISK, GPU_LINE, GPU_TRIANGLE
```

```python
# Python
process.setFluxEngineType(vps.FluxEngineType.AUTO)  # default
# or: CPU_DISK, GPU_DISK, GPU_LINE, GPU_TRIANGLE
```

`AUTO` chooses CPU or GPU based on the build and whether the selected model has a GPU implementation.

---

## Single-pass flux calculation

```c++
SmartPointer<viennals::Mesh<NumericType>> calculateFlux()
```

Computes flux for the current configuration and returns a mesh with flux data.

---

## Member functions

### Constructors

```c++
// Default
Process()

// From domain
Process(SmartPointer<Domain<NumericType, D>> passedDomain)

// From domain, model, and duration
template <typename ProcessModelType>
Process(SmartPointer<Domain<NumericType, D>> passedDomain,
        SmartPointer<ProcessModelType> passedProcessModel,
        const NumericType passedDuration = 0.)
```

### Set the domain

```c++
void setDomain(SmartPointer<Domain<NumericType, D>> passedDomain)
```

### Set the process model

```c++
void setProcessModel(SmartPointer<ProcessModel<NumericType, D>> passedProcessModel)
```

### Set the process duration

```c++
void setProcessDuration(NumericType passedDuration)
```

### Set parameters (unified)

```c++
void setParameters(const AdvectionParameters<NumericType>&)
void setParameters(const RayTracingParameters<NumericType, D>&)
void setParameters(const CoverageParameters<NumericType>&)
void setParameters(const AtomicLayerProcessParameters<NumericType>&)
```

### Set flux engine type
        
```c++
void setFluxEngineType(FluxEngineType type)
```

### Set intermediate output path
Path for writing intermediate results, if enabled. See [Logging]({% link misc/logging.md %}). for details.
```c++
void setIntermediateOutputPath(const std::string &path)
```

### Run the process

```c++
void apply()
```

