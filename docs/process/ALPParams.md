---
layout: default
title: Atomic Layer Process Parameters
parent: Running a Process
nav_order: 4
---

# Atomic Layer Process Parameters
{: .fs-9 .fw-500 }

---

`AtomicLayerProcessParameters` are specific to atomic-layer style process models (ALD/ALE-like), where the model behavior is organized into repeated cycles with one or more pulses (and optional purge steps). These parameters define the cycle count and the time discretization used for coverage evolution during pulses.

| Field              | Type          | Default | Description           |
| ------------------ | ------------- | ------- | --------------------- |
| `numCycles`        | `unsigned`    | `1`     | Number of ALP cycles to run. |
| `pulseTime`        | `double`      | `1.0`   | Duration of the active pulse step within a cycle. |
| `coverageTimeStep` | `double`      | `1.0`   | Time step used for updating coverages within a pulse (smaller can improve accuracy but increases runtime). |
| `purgePulseTime`   | `double`      | `0.0`   | Optional purge (or idle) duration after a pulse. Use `0.0` to disable. |