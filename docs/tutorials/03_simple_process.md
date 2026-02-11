---
layout: default
title: Simple Processes
parent: Tutorials
nav_order: 3
---

# Simple Processes

In ViennaPS, a simulation step involves applying a `Model` to a `Domain` for a specific duration.

## The Isotropic Process

The simplest model is `IsotropicProcess`. It moves the surface at a constant speed in the normal direction.
*   **Positive Rate**: Deposition (growth).
*   **Negative Rate**: Etching (removal).

```python
# Deposition: Grow material uniformly
depo_model = ps.IsotropicProcess(rate=2.0)

# Etching: Remove material uniformly
etch_model = ps.IsotropicProcess(rate=-1.5)
```

## Running a Process

To run the simulation, create a `ps.Process` object.

```python
# 1. Setup Process
process = ps.Process(domain, depo_model)

# 2. Set Duration
process.setProcessDuration(5.0)  # Run for 5 seconds (or arbitrary time units)

# 3. Apply
process.apply()
```

You can also run a process in multiple steps, for example to save snapshots.

```python
process.setProcessDuration(1.0)

for i in range(5):
    print(f"Running step {i}")
    process.apply()
    domain.saveHullMesh(f"step_{i}.vtp")
```
