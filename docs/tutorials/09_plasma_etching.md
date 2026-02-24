---
layout: default
title: SF₆/O₂ Plasma Etching
parent: Tutorials
nav_order: 9
---

# 2D SF₆/O₂ Plasma Etching with `SF6O2Etching`

This tutorial runs a **2D physical plasma etching model** using `SF6O2Etching`. It covers:

1. Create a masked trench geometry
2. Set physical units (required for physical models)
3. Configure model parameters (fluxes, ion energy distribution)
4. Configure advection and ray tracing settings
5. Configure coverage steady-state initialization
6. Enable logging and run the process
7. Save results

Download Jupyter Notebook: [09_plasma_etching.ipynb](https://raw.githubusercontent.com/ViennaTools/ViennaPS/refs/heads/documentation/examples/tutorials/09_plasma_etching.ipynb)

---

## 1. Import ViennaPS

```python
import viennaps as ps
```

---

## 2. Helper: Create a Masked Trench Domain

We use a helper function so the geometry setup stays in one place.

```python
def createTrenchMask():
    extent = 30
    gridDelta = 0.3

    domain = ps.Domain(xExtent=extent, gridDelta=gridDelta)

    ps.MakeTrench(
        domain,
        trenchWidth=10.0,
        trenchDepth=0.0,
        maskHeight=10.0,
        maskTaperAngle=10
    ).apply()

    return domain
```

### What this creates

* 2D domain with width `30` and resolution `0.3`
* A trench opening of width `10`
* Flat bottom (`trenchDepth=0.0`) so the focus is on mask + surface evolution
* A thick mask (`maskHeight=10.0`) with a small taper (`maskTaperAngle=10°`)

---

## 3. Set Units

Physical plasma models require consistent units.

```python
ps.Length.setUnit("um")  # micrometers
ps.Time.setUnit("min")   # minutes
```

This means:

* All geometry dimensions are interpreted in **µm**
* All process durations are interpreted in **minutes**

---

## 4. Set Up `SF6O2Etching` Model Parameters

Start from the default parameter set and override the key values.

```python
params = ps.SF6O2Etching.defaultParameters()

params.ionFlux = 10.0
params.etchantFlux = 4500.0
params.passivationFlux = 800.0

params.Ions.meanEnergy = 100.0
params.Ions.sigmaEnergy = 10.0
params.Ions.exponent = 500

model = ps.SF6O2Etching(params)
```

### What these parameters mean (high level)

* `ionFlux`: flux of ions that cause physical sputtering and ion-enhanced etching
* `etchantFlux`: flux of reactive species that chemically etch the surface
* `passivationFlux`: flux responsible for passivation

Ion energy distribution:

* `meanEnergy`: average ion energy
* `sigmaEnergy`: energy spread (Gaussian distribution width)
* `exponent`: controls angular spread in the initial ion directions (higher values → more directional)

The resulting `model` computes local etch/passivation behavior based on:

* ray-traced fluxes
* local surface orientation
* evolving surface coverages (reaction state)

---

## 5. Configure Advection and Ray Tracing Settings

### Advection (surface evolution scheme)

```python
processParams = ps.AdvectionParameters()
processParams.integrationScheme = ps.IntegrationScheme.LOCAL_LAX_FRIEDRICHS_1ST_ORDER
```

This selects a robust first-order scheme for the level-set advection.

### Ray tracing (flux sampling quality)

```python
rayTracingParams = ps.RayTracingParameters()
rayTracingParams.raysPerPoint = 1000
```

* `raysPerPoint` controls Monte Carlo sampling noise vs. runtime
* Higher values reduce noise but increase compute cost

---

## 6. Assemble the Process

```python
domain = createTrenchMask()
domain.saveVolumeMesh("SF6O2Etching_1")

process = ps.Process()
process.setDomain(domain)
process.setProcessModel(model)
process.setProcessDuration(10)  # 10 minutes
process.setParameters(processParams)
process.setParameters(rayTracingParams)
```

At this point you have:

* geometry (`domain`)
* physical etch model (`model`)
* numerical settings (advection + ray tracing)
* a fixed process duration of **10 min**

---

## 7. Coverage Steady-State Initialization

Physical plasma models often include **surface coverages** (e.g., passivation fraction, reactive coverage).
Before advancing the surface, the coverages are initialized to a steady state so reaction rates are stable.

```python
coverageParams = ps.CoverageParameters()
coverageParams.tolerance = 1e-4
coverageParams.maxIterations = 10

process.setParameters(coverageParams)
```

### Meaning

* `tolerance`: convergence criterion for relative coverage change
* `maxIterations`: hard cap on iterations if convergence is slow

So the solver will:

* iterate coverages until changes are small enough, or
* stop after 10 iterations even if the threshold is not reached

---

## 8. Enable Logging (Optional)

```python
ps.Logger.setLogLevel(ps.LogLevel.INTERMEDIATE)
```

This enables additional intermediate output during the run.
Useful for debugging convergence and checking that the model is behaving as expected.

---

## 9. Run the Process and Save Result

```python
process.apply()

domain.saveVolumeMesh("SF6O2Etching_2")
domain.show()
```

* `process.apply()` performs the etch simulation for the configured duration
* `saveVolumeMesh()` exports the final state to VTU for visualization

---

Download Jupyter Notebook: [09_plasma_etching.ipynb](https://raw.githubusercontent.com/ViennaTools/ViennaPS/refs/heads/documentation/examples/tutorials/09_plasma_etching.ipynb)



