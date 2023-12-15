---
layout: default
title: Plane
parent: Basic Geometries
grand_parent: Creating a Geometry
nav_order: 1
---

# Plane

---

This class provides a simple way to create a plane in a level set. It can be used to
create a substrate of any material. The plane can be added to an already existing
geometry or a new geometry can be created. The plane is created with normal
direction in the positive z direction in 3D and positive y direction in 2D. The plane
is centered around the origin with the total specified extent and height. The plane
can have a periodic boundary in the x and y (only 3D) direction.

```c++
// New geometry
psMakePlane(psDomainType domain, 
            const NumericType gridDelta,
            const NumericType xExtent, 
            const NumericType yExtent,
            const NumericType height, 
            const bool periodicBoundary = false,
            const psMaterial material = psMaterial::None)

// Add to existing geometry
psMakePlane(psDomainType domain, NumericType height = 0.,
            const psMaterial material = psMaterial::None)
```

Depending on which constructor for the plane-builder is called, the domain is either cleared and a new plane is inserted, or the plane is added to the existing geometry in the domain. A description of the parameters follows:

<dl>
<dt>domain</dt>
<dd>The `psDomain` object passed in a smart pointer.</dd>
<dt>gridDelta</dt>
<dd>Represents the grid spacing or resolution used in the simulation.</dd>
<dt>xExtent</dt>
<dd>Defines the extent of the plane geometry in the x-direction.</dd>
<dt>yExtent</dt>
<dd>Defines the extent of the plane geometry in the y-direction.</dd>
<dt>height</dt>
<dd>Sets the position of the plane in y(2D)/z(3D) direction.</dd>
<dt>periodicBoundary</dt>
<dd>(Optional) If set to true, enables periodic boundaries in both x and y directions. Default is set to false.</dd>
<dt>material</dt>
<dd>(Optional) Specifies the material used for the plane. Default is set to `psMaterial::None`.</dd>
</dl>

__Example usage__:

1. Creating a new domain: 

C++:
{: .label .label-blue }
```c++
auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
psMakePlane<NumericType, D>(domain, 0.5, 10.0, 10.0, 0.0, false,
                            psMaterial::Mask)
    .apply();
```
Python:
{: .label .label-green }
```python
domain = vps.Domain()
vps.MakePlane(domain=domain,
              gridDelta=0.5,
              xExtent=10.0,
              yExtent=10.0,
              height=0.0,
              periodicBoundary=False,
              material=vps.Material.Si,
             )
```

2. Adding plane to existing domain

C++: 
{: .label .label-blue }
```c++
auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
psMakePlane<NumericType, D>(domain, 0.5, 10.0, 10.0, 0.0, false,
                            psMaterial::Mask)
    .apply();
```

Python:
{: .label .label-green }
```python
domain = vps.Domain()
vps.MakePlane(domain=domain,
              gridDelta=0.5,
              xExtent=10.0,
              yExtent=10.0,
              height=0.0,
              periodicBoundary=False,
              material=vps.Material.Si,
             )
```