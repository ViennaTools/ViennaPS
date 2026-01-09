#include <geometries/psGeometryFactory.hpp>
#include <geometries/psMakeFin.hpp>
#include <geometries/psMakeHole.hpp>
#include <geometries/psMakePlane.hpp>
#include <geometries/psMakeStack.hpp>
#include <geometries/psMakeTrench.hpp>

namespace viennaps {

// Precompile specializations for geometries
PRECOMPILE_SPECIALIZE(GeometryFactory)
PRECOMPILE_SPECIALIZE(MakeFin)
PRECOMPILE_SPECIALIZE(MakeHole)
PRECOMPILE_SPECIALIZE(MakePlane)
PRECOMPILE_SPECIALIZE(MakeStack)
PRECOMPILE_SPECIALIZE(MakeTrench)

} // namespace viennaps