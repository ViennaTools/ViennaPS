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

The `psLogger` class can be used to set the verbosity of the program. The verbosity levels are:

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
psLogger::setLogLevel(psLogLevel::INTERMEDIATE);
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