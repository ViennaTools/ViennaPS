---
layout: default
title: Logging
parent: Miscellaneous
nav_order: 3
---

# Logging
{: .fs-9 .fw-500}

---

Singleton class for thread-safe logging.

The `Logger` class can be used to set the verbosity of the program. The verbosity levels are:

| Code           | Description                                     |
|----------------|-------------------------------------------------|
| `ERROR`        | Log errors only                                 |
| `WARNING`      | Log warnings                                    |
| `INFO`         | Log information (e.g. remaining time)           |
| `TIMING`       | Log timing results for the different methods    |
| `INTERMEDIATE` | Save intermediate results (disk meshes) during the process |
| `DEBUG`        | Debug                                           |

**Example usage:** Set the log level of the current program to `INTERMEDIATE`

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
```cpp
ps::Logger::setLogLevel(ps::LogLevel::INTERMEDIATE);
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green}
</summary>
```python
vps.Logger.setLogLevel(vps.LogLevel.INTERMEDIATE)
```
</details>

## Plasma etch diagnostics

At `INFO` verbosity or higher, the plasma models based on `PlasmaEtching`
(including SF6/O2, HBr/O2, and SF6/C4F8) and `FluorocarbonEtching` report the
normalized chemical, ion-enhanced, and sputtering contributions at the end of
a process. These are fractions of accumulated rate contributions, not rates
in length/time. The plasma base models additionally report substrate etch
depth in the configured length unit.

At `INTERMEDIATE` verbosity, flux processes write disk meshes containing
available flux, coverage, and surface-data arrays in cell data. Set the output
directory with `process.setIntermediateOutputPath(...)`. Source flux and
surface-emitted/desorption flux are distinct fields; the latter use the
`_surface` suffix.
