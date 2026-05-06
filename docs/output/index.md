---
layout: default
title: Geometry Output
nav_order: 9
has_children: true
---

# Geometry Output
{: .fs-9 .fw-700 }

---

To inspect the domain, a lightweight visualization method is provided via the `show()` member function. This function utilizes [VTK](https://vtk.org/) to render the current state of the simulation domain in an interactive window. It offers a quick and convenient way to visualize the geometry without the need for exporting files or using external visualization tools.

For advanced interactive rendering, the `VTKRenderWindow` class exposes camera, render-mode, screenshot, and multi-domain controls.

ViennaPS provides various methods for outputting the processed domain. The data is typically saved in the [VTK](https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html) file format, with surfaces, hulls, disks, and Level-Set grids stored in `.vtp` files and volume visualizations stored in `.vtu` files. For visualization, we recommend using [ParaView](https://www.paraview.org/), a powerful open-source visualization tool. Below, you'll find further details on the available geometry outputs.

Raw Level-Sets can also be written to `.lvst` files for later reuse with ViennaLS readers. Complete ViennaPS domains can be serialized with `Writer` and restored with `Reader` using the `.vpsd` domain format.

Common domain output functions include:

* `saveSurfaceMesh` / `getSurfaceMesh` for triangulated surfaces and interfaces.
* `saveHullMesh` / `getHullMesh` for hull meshes.
* `saveDiskMesh` / `getDiskMesh` for disk meshes containing material IDs.
* `saveVolumeMesh` for volume visualization meshes.
* `saveLevelSetMesh` / `getLevelSetMesh` for VTK Level-Set grid meshes.
* `saveLevelSets` for raw `.lvst` Level-Set files.
* `Writer` / `Reader` for full-domain `.vpsd` serialization.
