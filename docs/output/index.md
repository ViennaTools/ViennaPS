---
layout: default
title: Geometry Output
nav_order: 9
has_children: true
---

# Geometry Output
{: .fs-9 .fw-700 }

---

ViennaPS provides various methods for outputting the surface or volume of the processed domain. The data is typically saved in the [VTK](https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html) file format, with surfaces stored in `.vtp` files and volumes in `.vtu` files. For visualization, we recommend using [ParaView](https://www.paraview.org/), a powerful open-source visualization tool. Below, you'll find further details on the available geometry outputs.

In addition to VTK file formats, ViennaPS provides the flexibility to store level sets directly in the proprietary `.lvst` format. This feature enables users to save intermediate states during the process, allowing for more detailed analysis and the ability to revisit specific simulation stages. 
