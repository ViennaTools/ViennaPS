---
layout: default
title: Member Functions
parent: Simulation Domain
nav_order: 3
---

# `Domain` Class Documentation
{: .fs-9 .fw-500 }

---

## Type Aliases

```cpp
using lsDomainType = SmartPointer<viennals::Domain<NumericType, D>>;
using lsDomainsType = std::vector<lsDomainType>;
using csDomainType = SmartPointer<viennacs::DenseCellSet<NumericType, D>>;
using MaterialMapType = SmartPointer<MaterialMap>;
using MetaDataType = std::unordered_map<std::string, std::vector<double>>;
using Setup = DomainSetup<D>;

static constexpr char materialIdsLabel[] = "MaterialIds";
```

These aliases are used by the member function signatures below.

---

## Constructors

```cpp
Domain()
explicit Domain(SmartPointer<Domain> domain)
explicit Domain(lsDomainType levelSet)
explicit Domain(lsDomainsType levelSets)
Domain(NumericType gridDelta, NumericType xExtent,
       BoundaryType boundary = BoundaryType::REFLECTIVE_BOUNDARY)
Domain(NumericType gridDelta, NumericType xExtent, NumericType yExtent = 0.0,
       BoundaryType boundary = BoundaryType::REFLECTIVE_BOUNDARY)
Domain(double bounds[2 * D], BoundaryType boundaryConditions[D],
       NumericType gridDelta = 1.0)
explicit Domain(const Setup &setup)

template <class... Args>
static auto New(Args &&...args)
```

| Constructor | Description |
|-------------|-------------|
| `Domain()` | Creates an empty domain. |
| `Domain(SmartPointer<Domain> domain)` | Creates a deep copy of another domain. |
| `Domain(lsDomainType levelSet)` | Initializes the domain from one ViennaLS Level-Set. |
| `Domain(lsDomainsType levelSets)` | Initializes the domain from multiple ViennaLS Level-Sets. |
| `Domain(gridDelta, xExtent, boundary)` | Sets up a domain with the default primary direction. |
| `Domain(gridDelta, xExtent, yExtent, boundary)` | Sets up a rectangular domain. In 2D, `yExtent` is ignored. |
| `Domain(bounds, boundaryConditions, gridDelta)` | Sets up a domain from explicit bounds and boundary conditions. |
| `Domain(const Setup &setup)` | Creates a domain from a `DomainSetup<D>`. |
| `New(...)` | Convenience factory returning a `SmartPointer<Domain>`. |

---

## Key Member Functions

### Domain Setup

```cpp
void setup(const Setup &setup);
void setup(NumericType gridDelta, NumericType xExtent,
           NumericType yExtent = 0.0,
           BoundaryType boundary = BoundaryType::REFLECTIVE_BOUNDARY);
```

Initialize or reconfigure the simulation domain.

---

### Metadata

```cpp
void enableMetaData(MetaDataLevel level = MetaDataLevel::PROCESS);
void disableMetaData();
auto getMetaDataLevel() const;

void addMetaData(const std::string &key, const std::vector<double> &values);
void addMetaData(const std::string &key, double value);
void addMetaData(const MetaDataType &metaData);
auto getMetaData() const;
void clearMetaData(bool clearDomainData = false);
```

Enable, disable, access, and modify domain metadata. Metadata is written to exported VTK files.

---

### Level-Set Management

```cpp
void insertNextLevelSetAsMaterial(lsDomainType levelSet,
                                  std::string materialName,
                                  bool wrapLowerLevelSet = true);
void insertNextLevelSetAsMaterial(lsDomainType levelSet,
                                  Material material,
                                  bool wrapLowerLevelSet = true);
void insertMask(lsDomainType mask, Material material = Material::Mask);
void duplicateTopLevelSet(Material material);
void duplicateTopLevelSet(const std::string &materialName);
void removeTopLevelSet();
void removeLevelSet(unsigned int idx, bool removeWrapped = true);
void removeMaterial(Material material);
```

Insert, duplicate, or remove Level-Sets. Materials are assigned during insertion. With `wrapLowerLevelSet = true`, the inserted Level-Set is unioned with the current top layer. `insertMask` inserts the mask at the bottom of the Level-Set stack and wraps all existing layers against it.

---

### Topology Operations

```cpp
void removeStrayPoints();
std::size_t getNumberOfComponents() const;
```

Analyze and clean up surface topology in the domain.

--- 

### Boolean Operations

```cpp
void applyBooleanOperation(lsDomainType levelSet,
                           viennals::BooleanOperationEnum operation,
                           bool applyToAll = true);
```

Apply a Boolean operation with another Level-Set. By default, the operation is applied to all Level-Sets; with `applyToAll = false`, only the top Level-Set is modified.

---

### Cell-Set Generation

```cpp
void generateCellSet(NumericType position, Material coverMaterial, bool isAboveSurface = false);
```

Convert Level-Set stack into a Cell-Set for volume process modeling.

---

### Material Mapping

```cpp
void setMaterialMap(const MaterialMapType &materialMap);
void setMaterial(unsigned int lsId, Material material);
auto getMaterialsInDomain() const;
auto getMaterialLevelSet(Material material) const;
```

Assign or update material information for Level-Sets, query materials, or extract a Level-Set for one material. `getMaterialLevelSet` returns an empty pointer if the material is not present.

---

### Data Accessors

```cpp
auto& getSurface() const;
auto& getLevelSets() const;
unsigned int getNumberOfLevelSets() const;
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
std::vector<SmartPointer<viennals::Mesh<NumericType>>>
getLevelSetMesh(int width = 1);

SmartPointer<viennals::Mesh<NumericType>>
getSurfaceMesh(bool addInterfaces = false, bool sharpCorners = false,
               double minNodeDistanceFactor = 0.01) const;

SmartPointer<viennals::Mesh<NumericType>>
getHullMesh(NumericType bottomExtension = 0.0,
            bool sharpCorners = false) const;

SmartPointer<viennals::Mesh<NumericType>> getDiskMesh() const;

void saveLevelSetMesh(const std::string &fileName, int width = 1);
void saveSurfaceMesh(const std::string &fileName,
                     bool addInterfaces = true,
                     bool sharpCorners = false,
                     double minNodeDistanceFactor = 0.01) const;
void saveHullMesh(const std::string &fileName,
                  NumericType bottomExtension = 0.0,
                  bool sharpCorners = false) const;
void saveDiskMesh(const std::string &fileName) const;
void saveVolumeMesh(const std::string &fileName,
                    double wrappingLayerEpsilon = 1e-2) const;
void saveLevelSets(const std::string &fileName) const;
void print(std::ostream &out = std::cout, bool hrle = false) const;
void show() const;
```

Create or save surface, hull, disk, volume, or raw Level-Set output. `saveLevelSetMesh` writes one `.vtp` file per Level-Set with the suffix `_layerX.vtp`, while `saveLevelSets` writes raw `.lvst` files with `_layerX.lvst`.

| Parameter | Description |
|-----------|-------------|
| `addInterfaces` | Include internal material interfaces in the surface mesh. |
| `sharpCorners` | Preserve sharp corners during surface, interface, or hull meshing. |
| `minNodeDistanceFactor` | Minimum node distance factor used for multi-surface interface meshing. |
| `bottomExtension` | Extends hull meshes downward by the given distance when positive. |
| `wrappingLayerEpsilon` | Controls wrapping-layer tolerance for volume mesh output. |
| `hrle` | Also print the underlying HRLE Level-Set data. |

---

### Utilities

```cpp
void deepCopy(SmartPointer<Domain> other);
void clear();
```

Clone another domain or clear all internal data. `clear()` removes all Level-Sets and metadata and resets the material map.

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
