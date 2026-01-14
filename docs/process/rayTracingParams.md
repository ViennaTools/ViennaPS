---
layout: default
title: Ray Tracing Parameters
parent: Running a Process
nav_order: 2
---

# Ray Tracing Parameters
{: .fs-9 .fw-500 }

---

`RayTracingParameters` control the ray-tracing based flux computation (used by ray-based flux engines such as disk/line/triangle variants, depending on build and model support). These parameters mostly trade off runtime vs. noise/accuracy.

{: .note }
Some parameters are only meaningful for CPU ray tracing (e.g., `ignoreFluxBoundaries`).

| Field                    | Type                | Default  | Description |
| ------------------------ | ------------------- | -------- | ----------- |
| `normalizationType`      | `NormalizationType` | `SOURCE` | Flux normalization mode. Common options are `SOURCE` (normalize to the source strength) or `MAX` (normalize by maximum flux). |
| `ignoreFluxBoundaries`   | `bool`              | `false`  | Ignore boundary conditions during tracing (CPU only). |
| `useRandomSeeds`         | `bool`              | `true`   | If `true`, seeds the RNG non-deterministically (useful for Monte-Carlo averaging between runs). |
| `rngSeed`                | `unsigned`          | `0`      | Fixed RNG seed for reproducible runs. Only used if `useRandomSeeds == false`. |
| `raysPerPoint`           | `unsigned`          | `1000`   | Number of rays launched per surface point. Higher values reduce Monte-Carlo noise but increase runtime. |
| `smoothingNeighbors`     | `int`               | `1`      | Optional post-trace flux smoothing neighborhood size. Use `0` to disable smoothing. |
| `diskRadius`             | `double`            | `0.0`    | Ray-launch disk radius. `0` means automatic selection. |
| `minNodeDistanceFactor`  | `double`            | `0.05`   | Factor of the grid spacing used to derive the minimum node distance for triangle mesh generation. Higher values produce coarser meshes (faster, less detailed). |
| `maxBoundaryHits`        | `unsigned`          | `1000`   | Maximum number of boundary interactions allowed per ray before termination (prevents infinite bounce loops). |
| `maxReflections`         | `unsigned`          | *(max `unsigned`)* | Maximum number of reflections allowed per ray. The default is effectively unlimited. |