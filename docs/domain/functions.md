---
layout: default
title: Member Functions
parent: Simulation Domain
nav_order: 3
---

# `Domain` Member Functions 
{: .fs-9 .fw-500 }

---

Coming soon
{: .label .label-yellow}

## Constructors
```c++
Domain()
Domain(SmartPointer<Domain> passedDomain)
Domain(SmartPointer<viennals::Domain<NumericType, D>> passedLevelSet, bool generateCellSet = false, 
       const NumericType passedCellSetDepth = 0., const bool passedCellSetPosition = false)
Domain(lsDomainsType passedLevelSets, bool generateCellSet = false, 
       const NumericType passedCellSetDepth = 0., const bool passedCellSetPosition = false)
```

## Member Functions

```c++
void deepCopy(SmartPointer<Domain> passedDomain)
void insertNextLevelSet(SmartPointer<viennals::Domain<NumericType, D>> passedLevelSet, bool wrapLowerLevelSet = true)
void insertNextLevelSetAsMaterial(SmartPointer<viennals::Domain<NumericType, D>> passedLevelSet, const Material material, 
                                  bool wrapLowerLevelSet = true)
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
auto getBoundingBox() const
void print() const
void saveLevelSetMesh(std::string name, int width = 1)
void saveSurfaceMesh(std::string name, bool addMaterialIds = true)
void saveVolumeMesh(std::string name)
void saveLevelSets(std::string fileName) const
void clear()
```
