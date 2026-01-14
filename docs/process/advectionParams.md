---
layout: default
title: Advection Parameters
parent: Running a Process
nav_order: 1
---

# Advection Parameters
{: .fs-9 .fw-500 }

---

`AdvectionParameters` control the level-set advection step (i.e., how the geometry is advanced in time based on the model-provided velocity field). These parameters mainly affect stability, accuracy, and runtime.

{: .note }
`integrationScheme` is a legacy name for `spatialScheme` and is deprecated.

| Field                        | Type             | Default                    | Description |
| ---------------------------- | ---------------- | -------------------------- | ----------- |
| `spatialScheme`              | `SpatialScheme`  | `ENGQUIST_OSHER_1ST_ORDER` | Spatial discretization scheme for the level-set advection. |
| `integrationScheme` *(deprecated)* | `SpatialScheme` | `ENGQUIST_OSHER_1ST_ORDER` | Legacy alias for `spatialScheme`. Use `spatialScheme` instead. |
| `temporalScheme`             | `TemporalScheme` | `FORWARD_EULER`            | Time integration scheme for advection. |
| `timeStepRatio`              | `double`         | `0.4999`                   | CFL ratio used to compute the time step from the maximum stable step. Values closer to `0.5` are larger/faster but less conservative. |
| `dissipationAlpha`           | `double`         | `1.0`                      | Lax–Friedrichs dissipation scaling. Higher values are more diffusive (more stable, less sharp). |
| `adaptiveTimeStepSubdivisions` | `unsigned`     | `20`                       | Subdivisions used when adaptive time stepping is enabled. Higher values can improve robustness near interfaces at increased cost. |
| `checkDissipation`           | `bool`           | `true`                     | Enable dissipation checks (helps avoid unstable time steps). |
| `velocityOutput`             | `bool`           | `false`                    | Write velocity output per step (useful for debugging/analysis; may increase I/O). |
| `ignoreVoids`                | `bool`           | `false`                    | Ignore void regions during advection (model-dependent usefulness). |
| `adaptiveTimeStepping`       | `bool`           | `false`                    | Enable adaptive time stepping when approaching material interfaces (primarily relevant for etching). |

## Enum values

`SpatialScheme` corresponds to `viennals::SpatialSchemeEnum` (common options include `ENGQUIST_OSHER_1ST_ORDER`, `ENGQUIST_OSHER_2ND_ORDER`, `LAX_FRIEDRICHS_1ST_ORDER`, `LAX_FRIEDRICHS_2ND_ORDER`, `WENO_5TH_ORDER`).

`TemporalScheme` corresponds to `viennals::TemporalSchemeEnum` (`FORWARD_EULER`, `RUNGE_KUTTA_2ND_ORDER`, `RUNGE_KUTTA_3RD_ORDER`).