---
layout: default
title: Domain Setup
parent: Simulation Domain
nav_order: 4
---

# Domain Setup
{: .fs-9 .fw-500 }

---

The `DomainSetup` class defines the geometric grid configuration for a simulation domain in ViennaPS. It stores bounds, boundary conditions, and grid resolution (`gridDelta`) and is used internally by the `Domain` class to initialize the underlying `hrleGrid`.

---

## Features

* Manage domain extents and resolution
* Configure boundary conditions
* Create and store an `hrle::Grid` based on bounds and resolution
* Validate and print domain setup parameters
* Support halving geometries for symmetry exploitation

---

## Constructors

```cpp
DomainSetup();
DomainSetup(double bounds[2 * D], BoundaryType boundaryCons[D], NumericType gridDelta);
DomainSetup(NumericType gridDelta, NumericType xExtent, NumericType yExtent, BoundaryType boundary);
```

* **Default constructor** initializes all bounds to zero with `INFINITE_BOUNDARY`.
* **Bounding box constructor** accepts explicit bounds and boundary conditions.
* **Extent constructor** simplifies setup by defining half-extents along the x and y axes and applies default or specified boundary types.

---

## Member Functions

### Accessors

```cpp
auto& grid() const;
NumericType gridDelta() const;
std::array<double, 2 * D> bounds() const;
std::array<BoundaryType, D> boundaryCons() const;
NumericType xExtent() const;
NumericType yExtent() const;
```

Access the internal grid, resolution, and geometric/boundary parameters.

---

### Geometry Modification

```cpp
void halveXAxis();
void halveYAxis();
```

Modify the domain to simulate only half the geometry (along x or y), useful for symmetry. These operations are not allowed if periodic boundaries are used.

---

### Initialization

```cpp
void init();
void init(viennahrle::Grid<D> grid);
```

Construct or update the internal HRLE grid from the configured bounds, resolution, and boundary types.

---

### Debugging

```cpp
void print() const;
```

Prints all configured parameters to `stdout`, including grid delta, extents, and boundary configuration.

---

## Example

```cpp
using Setup = viennaps::DomainSetup<double, 3>;
BoundaryType boundaries[3] = {
    BoundaryType::REFLECTIVE_BOUNDARY,
    BoundaryType::REFLECTIVE_BOUNDARY,
    BoundaryType::INFINITE_BOUNDARY
};
double bounds[6] = {-5.0, 5.0, -5.0, 5.0, -1.0, 1.0};

Setup setup(bounds, boundaries, 0.5);
setup.print();
```

---

## Notes

* In 2D, the `yExtent()` function still exists but returns the fixed height of the domain.
* Grid cells are computed as integer multiples of `gridDelta`, ensuring alignment with HRLE.
* The final z-direction always has an `INFINITE_BOUNDARY` to prevent undesired reflection artifacts during etching or deposition.


