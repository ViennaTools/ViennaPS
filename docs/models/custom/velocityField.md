---
layout: default
title: Velocity Field
parent: Custom Models
grand_parent: Process Models
nav_order: 2
---

# Velocity Field
{: .fs-9 .fw-500}

---

Coming soon
{: .label .label-yellow}

## Surface Velocity Extension

| Option | Description                                      |
|--------|--------------------------------------------------|
|   0    | Do not translate level set ID to surface ID. This should be enabled if the surface velocity is only provided in the `psVelocityField` class and not through the `psSurfaceModel` class.     |
|   1    | Use unordered map to translate level set ID to surface ID. |
|   2    | Use KD-tree to translate level set ID to surface ID. The KD-tree uses a nearest neighbor lookup to determine the closest surface point and according velocity. |
