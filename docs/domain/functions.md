---
layout: default
title: Member Functions
parent: Simulation Domain
nav_order: 3
---

# `Domain` Class Documentation
{: .fs-9 .fw-500 }

---

## Constructors

* `Domain()` – Default constructor
* `Domain(SmartPointer<Domain>)` – Deep copy from another domain
* `Domain(lsDomainType)` – Initialize from a single Level-Set
* `Domain(lsDomainsType)` – Initialize from multiple Level-Sets
* `Domain(gridDelta, xExtent, yExtent = 0.0, boundary)` – Rectangular grid initialization
* `Domain(bounds[], boundaryConditions[], gridDelta)` – Bounding box initialization
* `Domain(const Setup &setup)` – From preconfigured domain setup

---

## Key Member Functions

### Domain Setup

```cpp
void setup(const Setup &setup);
void setup(NumericType gridDelta, NumericType xExtent, NumericType yExtent = 0.0, BoundaryType boundary);
```

Initialize or reconfigure the simulation domain.

---

### Level-Set Management

```cpp
void insertNextLevelSet(lsDomainType levelSet, bool wrap = true);
void insertNextLevelSetAsMaterial(lsDomainType levelSet, Material material, bool wrap = true);
void duplicateTopLevelSet(Material material);
void removeTopLevelSet();
void removeLevelSet(unsigned int idx, bool removeWrapped = true);
void removeMaterial(Material material);
void removeStrayPoints();
```

Insert, duplicate, or remove Level-Sets. Materials can be assigned during insertion.

---

### Boolean Operations

```cpp
void applyBooleanOperation(lsDomainType levelSet, viennals::BooleanOperationEnum op);
```

Apply a Boolean operation (e.g., union, intersection) across all Level-Sets in the domain.

---

### Cell-Set Generation

```cpp
void generateCellSet(NumericType position, Material coverMaterial, bool isAboveSurface = false);
```

Convert Level-Set stack into a Cell-Set for volume process modeling.

---

### Material Mapping

```cpp
void setMaterialMap(materialMapType map);
void setMaterial(unsigned int lsId, Material material);
```

Assign or update material information for Level-Sets.

---

### Data Accessors

```cpp
auto& getSurface() const;
auto& getLevelSets() const;
auto& getMaterialMap() const;
auto& getCellSet() const;
auto& getGrid() const;
auto getGridDelta() const;
auto& getSetup();
auto getBoundingBox() const;
auto getBoundaryConditions() const;
```

Access the surface Level-Set, grid, bounding box, material map, and setup.

---

### Output & Export

```cpp
void saveLevelSetMesh(std::string fileName, int width = 1);
void saveSurfaceMesh(std::string fileName, bool addInterfaces = true,
                     double wrappingLayerEpsilon = 0.01, bool boolMaterials = false);
auto getSurfaceMesh(bool addInterfaces = false, double wrappingLayerEpsilon = 0.01,
                    bool boolMaterials = false);
void saveVolumeMesh(std::string fileName, double wrappingEps = 1e-2) const;
void saveHullMesh(std::string fileName, double wrappingEps = 1e-2) const;
void saveLevelSets(std::string prefix) const;
void print() const;
```

Save surface or volume meshes in VTK formats or print domain state to `stdout`.

---

### Utilities

```cpp
void deepCopy(SmartPointer<Domain> other);
void clear();
```

Clone another domain or clear all internal data.

---

## Notes

* Level-Sets inserted with `wrap = true` are merged with underlying layers.
* Material IDs are stored under the `"MaterialIds"` label in exported meshes.
* Cell-Sets are not automatically generated and must be explicitly created using `generateCellSet`.

---

## Example Usage

```cpp
using DomainType = viennaps::Domain<double, 3>;
auto domain = DomainType::New(1.0, 10.0, 10.0);
auto ls = viennals::Domain<double, 3>::New();
// setup ls...
domain->insertNextLevelSetAsMaterial(ls, Material::Si);
domain->saveSurfaceMesh("surface.vtp");
```

---

## See Also

* [ViennaLS Documentation](https://viennatools.github.io/ViennaLS/)

