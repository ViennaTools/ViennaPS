---
layout: default
title: Geometry Output
nav_order: 9
has_children: true
---

# Geometry Output
{: .fs-9 .fw-700 }

---

To inspect the domain, a leightweight visualization method is provided via the `show()` member function. This function utilizes [VTK](https://vtk.org/) to render the current state of the simulation domain in an interactive window. It offers a quick and convenient way to visualize the geometry without the need for exporting files or using external visualization tools.

ViennaPS provides various methods for outputting the surface or volume of the processed domain. The data is typically saved in the [VTK](https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html) file format, with surfaces stored in `.vtp` files and volumes in `.vtu` files. For visualization, we recommend using [ParaView](https://www.paraview.org/), a powerful open-source visualization tool. Below, you'll find further details on the available geometry outputs.

In addition to VTK file formats, ViennaPS provides the flexibility to domain directly in the proprietary `.vpsd` format. This feature enables users to save intermediate states during the process, allowing for more detailed analysis and the ability to revisit specific simulation stages. 
