---
layout: default
title: Advection Callback
parent: Custom Models
grand_parent: Process Models
nav_order: 5
---

# Advection Callback
{: .fs-9 .fw-500}

---

An `AdvectionCallback<NumericType, D>` runs user code around each advection
step. Attach it to a process model with `setAdvectionCallback(...)`; the process
sets its `domain` member before execution.

```c++
#include <process/psAdvectionCallback.hpp>

template <typename T, int D>
class SaveSurface : public viennaps::AdvectionCallback<T, D> {
public:
  bool applyPostAdvect(T processTime) override {
    this->domain->saveSurfaceMesh("latest_surface.vtp");
    return true;
  }
};
```

With an existing model:

```c++
model->setAdvectionCallback(
    viennaps::SmartPointer<SaveSurface<double, 2>>::New());
```

The same callback can be implemented in Python:

```python
import viennaps as vps

class SaveSurface(vps.AdvectionCallback):
    def __init__(self):
        super().__init__()

    def applyPostAdvect(self, process_time):
        self.domain.saveSurfaceMesh("latest_surface.vtp")
        return True

callback = SaveSurface()
model.setAdvectionCallback(callback)
```

Keep the Python callback alive while the process runs. These examples overwrite
one snapshot; use distinct filenames to retain a time series.

| Hook | When it runs in analytic and flux processes |
|------|---------------------------------------------|
| `applyPreAdvect(processTime)` | Before advection, with the elapsed process time. |
| `applyPostAdvect(processTime)` | After advection, with the updated elapsed process time. |

Both hooks return a boolean. Return `true` to continue or `false` to stop an
analytic or flux process early. The post-advection argument is cumulative
process time, despite the `advectionTime` name in the base-class declaration.

## Callback-only execution

For a model with a callback and a `Process` duration of zero, the callback-only
strategy invokes `applyPreAdvect(0)` once. It does not advect the surface or
invoke `applyPostAdvect`, and it does not use the returned boolean to set the
process result. This is how
[ion implantation]({% link models/prebuilt/ionImplantation.md %}) and
[annealing]({% link models/prebuilt/anneal.md %}) run volume calculations. Those
models already own a callback; replacing it replaces their volume operation.
