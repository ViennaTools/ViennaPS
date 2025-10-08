---
layout: default
title: General Notes
nav_order: 4
---

# General Notes
{: .fs-9 .fw-700}

---

## Switching between 2D and 3D

In the C++ version of ViennaPS, the simulation dimensionality is determined **at compile time**. It is defined as the **second template parameter** in most classes and functions across the framework.
    
For example:

```cpp
ps::Domain<double, 3> domain3D;  // 3D simulation
ps::Domain<double, 2> domain2D;  // 2D simulation
```

Because the dimensionality is a compile-time parameter, **2D and 3D classes cannot be mixed** within the same simulation. Once chosen, the dimensionality remains fixed for all components of that simulation.

For users who need to transition from a 2D to a 3D simulation, ViennaPS offers the [Extrude]({% link misc/extrusion.md %}) utility. This utility enables the extrusion of a 2D domain to 3D, providing a seamless way to extend simulations across different dimensions.

In the Python bindings, the simulation dimensionality is organized into two dedicated modules: **`d2`** for 2D and **`d3`** for 3D simulations. Users can choose the appropriate module depending on their needs.

By default, ViennaPS operates in **2D mode**. To switch to 3D, call:

```python
ps.setDimension(3)
```

Alternatively, the dimensional modules can be accessed explicitly:

```python
ps.d2.Domain   # 2D domain
ps.d3.Domain   # 3D domain
```


## Namespace

ViennaPS is encapsulated within the `viennaps` namespace. This design choice ensures that all classes, functions, and utilities within the library are organized under a single namespace, providing a clean and structured interface for users. When working with ViennaPS, it is essential to include the `viennaps` namespace in your code to access the library's functionality. In this documentation, the namespace is omitted for brevity, but it should be included in your code.

## Numeric Types (C++ only)

ViennaPS supports the utilization of either `float` or `double` as the underlying numeric type. While `float` might offer slightly higher performance in some cases, it is generally recommended to use `double` in your simulation due to its enhanced precision.

It's essential to note that the choice of numeric type is a static (compile-time) parameter in every ViennaPS class and function. Once a numeric type is selected for a particular simulation, it is not possible to switch to a different numeric type within the program.

Additionally, for users working with Python bindings, it's important to be aware that the Python interface always uses `double` as the numeric type.


## Using Smart Pointers (C++ only)

In ViennaPS, smart pointers are utilized to pass domains, models, and other essential objects to processes and utility functions. To facilitate this, the library includes a custom class named `SmartPointer`, serving as a shared pointer implementation. This design choice ensures efficient memory management and enables seamless interaction between different components within the simulation framework.

__Example:__

```c++
using namespace viennaps;

// Creating a new domain
auto domain = SmartPointer<Domain<NumericType, D>>::New();
auto domain = Domain<NumericType, D>::New(); // Shorter syntax but does the same

// Using a pre-built model
auto model = SmartPointer<IsotropicProcess<NumericType, D>>::New(/*pass constructor arguments*/);
```