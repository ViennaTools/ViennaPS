#pragma once

#include <array>
#include <atomic>
#include <vector>

// TEST (LS_IONNORM, LS_SAMESURF_DEG, 2026-10-18/19): the surface-element
// normals averaged over a radius, filled by CPUTriangleEngine. ChemicalIon
// reads them for the ion's incidence angle and reflection, ElementToPointData
// for averaging the fluxes only over elements of the same surface. Empty
// unless LS_IONNORM is set.
namespace viennaps::ionnorm {
inline std::vector<std::array<float, 3>> averaged;
inline std::atomic<unsigned long long> hits{0}, facet{0}, stopped{0};
} // namespace viennaps::ionnorm
