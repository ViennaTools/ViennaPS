---
layout: default
title: Member Functions
parent: Simulation Domain
nav_order: 3
---

# `psDomain` Member Functions 
{: .fs-9 .fw-500 }

---

Coming soon
{: .label .label-yellow}

## Constructors
```c++
psDomain()
psDomain(psSmartPointer<psDomain> passedDomain)
psDomain(lsDomainType passedLevelSet, bool generateCellSet = false, 
         const NumericType passedCellSetDepth = 0., const bool passedCellSetPosition = false)
psDomain(lsDomainsType passedLevelSets, bool generateCellSet = false, 
         const NumericType passedCellSetDepth = 0., const bool passedCellSetPosition = false)
```

## Member Functions

```c++
void deepCopy(psSmartPointer<psDomain> passedDomain)
void insertNextLevelSet(lsDomainType passedLevelSet, bool wrapLowerLevelSet = true)
void insertNextLevelSetAsMaterial(lsDomainType passedLevelSet, const psMaterial material, 
                                  bool wrapLowerLevelSet = true)
void duplicateTopLevelSet(const psMaterial material = psMaterial::None)
void removeTopLevelSet()
void applyBooleanOperation(lsDomainType levelSet, lsBooleanOperationEnum operation)
void generateCellSet(const NumericType depth = 0., const bool passedCellSetPosition = false)
void setMaterial(unsigned int lsId, const psMaterial material)
auto &getLevelSets() const
auto &getMaterialMap() const
auto &getCellSet() const
auto &getGrid() const
void print() const
void saveLevelSetMesh(std::string name, int width = 1)
void saveSurfaceMesh(std::string name, bool addMaterialIds = true)
void saveVolumeMesh(std::string name)
void saveLevelSets(std::string fileName) const
void clear()
```
