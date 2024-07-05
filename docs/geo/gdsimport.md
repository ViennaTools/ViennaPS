---
layout: default
title: GDS File Import
parent: Creating a Geometry
nav_order: 2
---

# Importing a GDSII Mask File
{: .fs-9 .fw-500 }

---

ViennaPS provides a convenient feature allowing users to import geometries directly from GDSII mask files. It's important to note that ViennaPS focuses on handling simple geometry information extracted from GDSII files, without supporting additional data that might be stored within the GDSII format.

To parse a GDSII file using ViennaPS, follow these steps:

1. **Create a `ps::GDSGeometry` Object:**
   Initialize a `ps::GDSGeometry` object, specifying the desired grid spacing, boundary conditions, and additional padding (optional).

    ```cpp
    auto mask = ps::SmartPointer<ps::GDSGeometry<NumericType, D>>::New();
    mask->setGridDelta(gridDelta);
    mask->setBoundaryConditions(boundaryConds);
    mask->setBoundaryPadding(xPad, yPad);
    ```

   Replace `gridDelta`, `boundaryConds`, and `xPad`,`yPad` with your preferred values. The geometry is always parsed on a plane normal to the z direction.
   The values of `xPad` and `yPad` are always added to the largest and subtracted from the smallest extension of all geometries in the GDSII file.

2. **Use `ps::GDSReader` to Parse Geometry:**
   Utilize the `ps::GDSReader` to parse the GDSII file into the previously created `ps::GDSGeometry` object.

    ```cpp
    ps::GDSReader<NumericType, D>(mask, "path/to/your/file.gds").apply();
    ```

   Replace `"path/to/your/file.gds"` with the actual path to your GDSII file.

3. **Convert Single Layers to Level Sets:**
   Extract specific layers from the parsed GDSII geometry, convert them into level sets, and add them to your simulation domain. To access a particular layer, provide its GDSII layer number.

    ```cpp
    // Create new domain
    auto domain = ps::SmartPointer<ps::Domain<NumericType, D>>::New(); 

    // Convert the layer to a level set and add it to the domain.
    auto layer = mask->layerToLevelSet(0 /*layer*/, 0 /*base z position*/, 0.5 /*height*/);
    domain->insertNextLevelSet(layer);
    ```

   Replace `layerNumber` with the GDSII layer number you wish to access.

   Layers can also be inverted to be used a mask.
    ```cpp
    // Create new domain
    auto domain = ps::SmartPointer<ps::Domain<NumericType, D>>::New(); 

    // Convert the inverted layer to a level set and add it to the domain.
    auto layer = mask->layerToLevelSet(0 /*layer*/, 0 /*base z position*/, 
                                       0.5 /*height*/, true /*invert*/);
    domain->insertNextLevelSetAsMaterial(layer, ps::Material::Mask);

    // Create substrate underneath the mask
    ps::MakePlane<NumericType, D>(domain, 0. /*base z position*/, ps::Material::Si).apply();
    ```

## Related Examples

* [GDS Reader](https://github.com/ViennaTools/ViennaPS/tree/master/examples/GDSReader)
