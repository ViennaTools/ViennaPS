---
layout: default
title: Member Functions
parent: Simulation Domain
nav_order: 3
---

# `Domain` Member Functions 
{: .fs-9 .fw-500 }

---

## Constructors
```c++
// namespace viennaps
Domain()
Domain(SmartPointer<Domain> passedDomain) // deep copy contructor
Domain(SmartPointer<viennals::Domain<NumericType, D>> passedLevelSet)
Domain(std::vector<viennals::Domain<NumericType, D>> passedLevelSets)

// Configure DomainSetup (v3.3.0) for basic geometry builders
Domain(NumericType gridDelta, NumericType xExtent,
       BoundaryType boundary = BoundaryType::REFLECTIVE_BOUNDARY) // 2D only
Domain(NumericType gridDelta, NumericType xExtent, NumericType yExtent = 0.0,
       BoundaryType boundary = BoundaryType::REFLECTIVE_BOUNDARY)
Domain(const Setup &setup)
```

## Member Functions

```c++
void setup(const Setup &setup) 
void setup(NumericType gridDelta, NumericType xExtent, NumericType yExtent = 0,
           BoundaryType boundary = BoundaryType::REFLECTIVE_BOUNDARY)
void deepCopy(SmartPointer<Domain> passedDomain)
void insertNextLevelSet(SmartPointer<viennals::Domain<NumericType, D>> passedLevelSet, 
                        bool wrapLowerLevelSet = true)
void insertNextLevelSetAsMaterial(SmartPointer<viennals::Domain<NumericType, D>> passedLevelSet, 
                                  const Material material, bool wrapLowerLevelSet = true)
void duplicateTopLevelSet(const Material material = Material::None)
void removeTopLevelSet()
void removeLevelSet(unsigned int idx, bool removeWrapped = true)
void removeMaterial(const Material material)
void applyBooleanOperation(SmartPointer<viennals::Domain<NumericType, D>> levelSet, viennals::BooleanOperationEnum operation)
void generateCellSet(const NumericType position, const Material coverMaterial,
                     const bool isAboveSurface = false)
void setMaterial(unsigned int lsId, const Material material)
auto &getLevelSets() const
auto &getMaterialMap() const
auto &getCellSet() const
auto &getGrid() const
auto getGridDelta() const
auto &getSetup()
auto getBoundingBox() const
auto getBoundaryConditions() const
void print() const
void saveLevelSetMesh(std::string name, int width = 1)
void saveSurfaceMesh(std::string name, bool addMaterialIds = true)
void saveVolumeMesh(std::string name, double wrappingLayerEpsilon = 1e-2)
void saveHullMesh(std::string name, double wrappingLayerEpsilon = 1e-2)
void saveLevelSets(std::string fileName) const
void clear()
```
