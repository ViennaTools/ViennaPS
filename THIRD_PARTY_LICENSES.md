# Third-Party Licenses

ViennaPS depends on several third-party libraries. These are **not bundled** with the source code and are retrieved automatically during build via CMake.

Each dependency is governed by its own license, listed below.

---

### ViennaCore

**License:** MIT
**URL:** [https://github.com/ViennaTools/viennacore](https://github.com/ViennaTools/viennacore)

---

### ViennaLS

**License:** MIT
**URL:** [https://github.com/ViennaTools/viennals](https://github.com/ViennaTools/viennals)

#### ViennaHRLE

**License:** MIT
**URL:** [https://github.com/ViennaTools/viennahrle](https://github.com/ViennaTools/viennahrle)

#### VTK (Visualization Toolkit)

**License:** BSD 3-Clause
**URL:** [https://vtk.org/](https://vtk.org/)
**Note:** Used for mesh export and visualization.

---

### ViennaRay

**License:** MIT
**URL:** [https://github.com/ViennaTools/viennaray](https://github.com/ViennaTools/viennaray)

#### Embree

**License:** Apache License 2.0
**URL:** [https://www.embree.org/](https://www.embree.org/)
**Note:** Used as CPU backend for ray tracing.

---

### ViennaCS

**License:** MIT
**URL:** [https://github.com/ViennaTools/viennacs](https://github.com/ViennaTools/viennacs)

---

### pybind11

**License:** BSD 3-Clause
**URL:** [https://github.com/pybind/pybind11](https://github.com/pybind/pybind11)
**Note:** Used only when building Python bindings.

---

### NVIDIA OptiX (Optional Dependency)

**License:** Proprietary (NVIDIA Software Developer Kits, Samples and Tools License Agreement)
**URL:** [https://developer.nvidia.com/nvidia-optix-license](https://developer.nvidia.com/nvidia-optix-license)

**Note:**
GPU acceleration via NVIDIA OptiX is **not included** with ViennaPS and must be obtained separately from NVIDIA.
Use of OptiX is subject to NVIDIA’s proprietary license, which is **not compatible with GPL-3.0-only**.
In particular:

* OptiX **must not be redistributed** as part of ViennaPS.
* ViennaPS builds with OptiX **must not be used in cloud services or hosted environments**.
* Users must **manually accept NVIDIA’s license** before enabling OptiX support.

OptiX support in ViennaPS is entirely optional and disabled by default.

---

## Notes

* All listed licenses (except OptiX) are **GPL-3.0-compatible**.
* ViennaPS includes only source code under GPL-3.0-only or compatible licenses.
* Optional dependencies such as OptiX are clearly marked and excluded from official source and binary distributions.

---
