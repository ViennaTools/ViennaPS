---
layout: default
title: Advection Parameters
parent: Running a Process
nav_order: 1
---

# Advection Parameters
{: .fs-9 .fw-500 }

---

| Field               | Type                    | Default                    | Description                        |
| ------------------- | ----------------------- | -------------------------- | ---------------------------------- |
| `spatialScheme`     | `SpatialScheme`         | `ENGQUIST_OSHER_1ST_ORDER` | Level-set spatial discretization scheme.      |
| `temporalScheme`    | `TemporalScheme`        | `FORWARD_EULER`            | Time integration scheme.           |
| `timeStepRatio`     | `double`                | `0.4999`                   | CFL ratio.                         |
| `dissipationAlpha`  | `double`                | `1.0`                      | Lax–Friedrichs dissipation factor. |
| `checkDissipation`  | `bool`                  | `true`                     | Enable dissipation check.          |
| `velocityOutput`    | `bool`                  | `false`                    | Write velocity per step.           |
| `ignoreVoids`       | `bool`                  | `false`                    | Ignore void regions.               |
| `adaptiveTimeStepping` | `bool`           | `false`                    | Enable adaptive time stepping when approaching material interfaces during etching.     |