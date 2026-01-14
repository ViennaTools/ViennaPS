---
layout: default
title: Ray Tracing Parameters
parent: Running a Process
nav_order: 2
---

# Ray Tracing Parameters
{: .fs-9 .fw-500 }

---

| Field                  | Type                | Default  | Description                        |
| ---------------------- | ------------------- | -------- | ---------------------------------- |
| `raysPerPoint`         | `unsigned`          | `1000`   | Rays per surface point.            |
| `smoothingNeighbors`   | `int`               | `1`      | Post-trace flux smoothing.         |
| `useRandomSeeds`       | `bool`              | `true`   | Random seeding.                    |
| `rngSeed`              | `unsigned`          | `0`      | Fixed seed for the RNG. (`useRandomSeeds` must be `false` to use this.)            |
| `ignoreFluxBoundaries` | `bool`              | `false`  | Ignore BCs in tracing (CPU only).  |
| `minNodeDistanceFactor`| `double`           | `0.05`  | Factor for triangle mesh generation. A higher factor creates a coarser mesh. |
| `diskRadius`           | `double`            | `0`      | Disk radius; `0` = auto.           |
| `normalizationType`    | `NormalizationType` | `SOURCE` | Normalization (`SOURCE` or `MAX`). |