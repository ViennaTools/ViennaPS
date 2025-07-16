---
layout: default
title: VTK Metadata Export
parent: Geometry Output
nav_order: 4
---

# VTK Metadata Export
{: .fs-9 .fw-500 }

---


ViennaPS supports exporting **metadata** to VTK output files. This feature allows users to embed additional simulation-specific and domain-specific information into the VTK files generated during surface, hull, or volume mesh output. The metadata is useful for post-processing, debugging, visualization, and reproducibility of simulation results.

Metadata can be selectively included in VTK output by setting a **metadata level**, which controls the amount and type of information written. Metadata export is configured at the **Domain** level using the following static method:

```cpp
Domain<NumericType, D>::enableMetaData(const MetaDataLevel level = MetaDataLevel::PROCESS);
```

Once enabled, metadata will be attached to all future VTK output from any domain instance during the current execution.

---

## Metadata Levels

The `MetaDataLevel` enum defines the available levels of metadata:

| Level     | Description                                                                                     |
| --------- | ----------------------------------------------------------------------------------------------- |
| `NONE`    | No metadata is written to VTK files.                                                            |
| `GRID`    | Domain-specific metadata only (e.g., grid spacing, boundary conditions).                        |
| `PROCESS` | Includes domain metadata and parameters from the **most recently applied process**.             |
| `FULL`    | Includes all available metadata, such as advection model settings, ray tracing parameters, etc. |

The default level when enabling metadata is `MetaDataLevel::PROCESS`.

---

## Usage Example

```cpp
// Enable metadata export with full detail
Domain<float, 3>::enableMetaData(MetaDataLevel::FULL);

// Create domain and apply process
Domain<float, 3> dom(...);
SF6O2EtchingProcess process;
process.apply(dom);

// Write output
domain.saveSurfaceMesh("output_surface.vtp"); // Surface mesh with metadata
```

The resulting VTK file will now contain metadata including grid spacing, boundary conditions, and all relevant parameters from the `SF6O2EtchingProcess`.

---

## Process Metadata Behavior

* Metadata at the `PROCESS` or `FULL` level always reflects the **most recently applied process** via `.apply()`.
* When a new process is applied to a domain, the previously stored process metadata is cleared and replaced with the current one.
* Only one set of process metadata is retained per domain at any time.

---

## Affected Output Types

Metadata is supported in the following mesh outputs:

* Surface mesh (`.vtp`)
* Hull mesh (`.vtp`)
* Volume mesh (`.vtu`)

Each writer automatically includes the metadata based on the current global setting.

---

## Notes

* The metadata system is **static** and global. Once enabled, it affects all domain instances during the simulation run.
* For deterministic and reproducible output, it is recommended to enable metadata at the start of the simulation.



