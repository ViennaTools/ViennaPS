---
layout: default
title: Point Data
parent: Miscellaneous
nav_order: 5
---

# Point Data
{: .fs-9 .fw-500}

---

The `viennals::PointData` class is designed to hold data associated with points in space. It's a generic class that can work with any data type `T` that satisfies the `lsConcepts::IsFloatingPoint` concept. By default, it uses `double` as the data type.

__Member Types:__

* `ScalarDataType`: A type alias for a vector of `T` type elements. This is used to represent scalar data.
* `VectorDataType`: A type alias for a vector of arrays, each containing 3 elements of `T` type. This is used to represent vector data.

__Public Methods:__

* `insertNextScalarData`: This method inserts a new scalar data its corresponding label.