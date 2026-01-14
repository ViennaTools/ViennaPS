---
layout: default
title: Coverage Parameters
parent: Running a Process
nav_order: 3
---

# Coverage Parameters
{: .fs-9 .fw-500 }

---

`CoverageParameters` are used by models that evolve one or more surface coverages (e.g., PlasmaEtching models, Fluorocarbon, and atomic-layer style models). They typically affect how the initial coverage state is computed and/or how iterative coverage updates converge.

{: .note }
`initialized` is an internal flag and is usually managed by the process/model implementation.

| Field           | Type       | Default              | Description |
| --------------- | ---------- | -------------------- | ----------- |
| `tolerance`     | `double`   | `0.0`                | Convergence threshold for iterative coverage initialization/updates. `0.0` typically disables tolerance-based early stopping (iteration limit only). |
| `maxIterations` | `unsigned` | *(max `unsigned`)*   | Maximum number of iterations for coverage initialization/updates. The default is effectively unlimited. |
| `initialized`   | `bool`     | `false`              | Internal bookkeeping flag indicating whether the coverage state was initialized. Usually you do not need to set this manually. |