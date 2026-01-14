# Third-Party Licenses

ViennaPS is licensed under the MIT License.  
This file lists third-party libraries used by ViennaPS and their respective licenses.

Some dependencies are optional (e.g., GPU acceleration via NVIDIA OptiX) and may be subject to additional terms.

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

**License:** Proprietary — NVIDIA Software Developer Kits, Samples and Tools License Agreement  
**URL:** https://developer.nvidia.com/designworks/sdk-samples-tools-software-license-agreement
**Notes:**  
OptiX is an optional GPU backend for ray tracing. It is **not included** in ViennaPS and must be obtained separately from NVIDIA.  
Use of OptiX is subject to **NVIDIA's proprietary license**.

If users choose to build ViennaPS with OptiX support:

- They must **manually download and install OptiX** from NVIDIA’s website.
- They must **accept NVIDIA’s license terms**.

ViennaPS builds on PyPI and official releases do **not include or link against OptiX**.

---

