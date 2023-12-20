---
layout: default
title: General Notes
nav_order: 4
---

# General Notes
{: .fs-9 .fw-700}

---

## Numeric Types

ViennaPS supports the utilization of either `float` or `double` as the underlying numeric type. While `float` might offer slightly higher performance in some cases, it is generally recommended to use `double` in your simulation due to its enhanced precision.

It's essential to note that the choice of numeric type is a static (compile-time) parameter in every ViennaPS class and function. Once a numeric type is selected for a particular simulation, it is not possible to switch to a different numeric type within the program.

Additionally, for users working with Python bindings, it's important to be aware that the Python interface always uses `double` as the numeric type.

## Switching between 2D and 3D mode

ViennaPS provides the flexibility for users to choose between 2D and 3D modes during compile time. The dimensionality is specified as a second template (static) parameter, and most classes and functions in ViennaPS adhere to this structure. It's important to note that 2D and 3D classes cannot be mixed within the same simulation, and the choice of dimensionality is fixed at compile time.

For users who need to transition from a 2D to a 3D simulation, ViennaPS offers the [psExtrude]({% link misc/extrusion.md %}) utility. This utility enables the extrusion of a 2D domain to 3D, providing a seamless way to extend simulations across different dimensions.

## Using Smart Pointers

Coming soon
{: .label .label-yellow}