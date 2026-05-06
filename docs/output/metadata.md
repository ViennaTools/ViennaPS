---
layout: default
title: VTK Metadata Export
parent: Geometry Output
nav_order: 6
---

# VTK Metadata Export
{: .fs-9 .fw-500 }

---


ViennaPS supports exporting **metadata** to VTK output files. This feature allows users to embed additional simulation-specific and domain-specific information into the VTK files generated during surface, hull, or volume mesh output. The metadata is useful for post-processing, debugging, visualization, and reproducibility of simulation results.

Metadata can be selectively included in VTK output by setting a **metadata level**, which controls the amount and type of information written. Metadata export is configured on each **Domain** instance:

```cpp
void enableMetaData(MetaDataLevel level = MetaDataLevel::PROCESS);
void disableMetaData();
auto getMetaDataLevel() const;

void addMetaData(const std::string &key, const std::vector<double> &values);
void addMetaData(const std::string &key, double value);
void addMetaData(const MetaDataType &metaData);
auto getMetaData() const;
void clearMetaData(bool clearDomainData = false);
```

Once enabled, metadata will be attached to future VTK output written by that domain.

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
// Create domain and apply process
auto domain = Domain<double, 3>::New();
domain->enableMetaData(MetaDataLevel::FULL);
...
Process<double, 3> process(domain, model);
process.apply();

// Write output
domain->saveSurfaceMesh("output_surface.vtp"); // Surface mesh with metadata
```

The resulting VTK file will now contain metadata including grid spacing, boundary conditions, and all relevant parameters from the applied process.

<details markdown="1">
<summary markdown="1">
Python:
{: .label .label-green }
</summary>
```python
# Create domain and apply process
domain = vps.Domain()
domain.enableMetaData(vps.MetaDataLevel.FULL)
...
process = vps.Process(domain, model)
process.apply()

# Write output
domain.saveSurfaceMesh("output_surface.vtp") # Surface mesh with metadata
```
</details>

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
* Disk mesh (`.vtp`)
* Level-Set mesh (`.vtp`)
* Volume mesh (`.vtu`)

Each writer includes the metadata stored on the domain.

---

## Notes

* For deterministic and reproducible output, it is recommended to enable metadata before creating geometry or applying processes.
* Metadata can also be added manually with `addMetaData`.
