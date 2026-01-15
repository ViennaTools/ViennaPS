#include <gds/psGDSGeometry.hpp>
#include <gds/psGDSReader.hpp>

#include <psDomain.hpp>
#include <psDomainSetup.hpp>
#include <psExtrude.hpp>
#include <psPlanarize.hpp>
#include <psRateGrid.hpp>
#include <psReader.hpp>
#include <psSlice.hpp>
#include <psToDiskMesh.hpp>
#include <psWriter.hpp>

#include <lsPreCompileMacros.hpp>

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

namespace viennaps {

// Precompile specializations for commonly used classes
PRECOMPILE_SPECIALIZE(Domain)
PRECOMPILE_SPECIALIZE(Planarize)
PRECOMPILE_SPECIALIZE(Reader)
PRECOMPILE_SPECIALIZE(Writer)
PRECOMPILE_SPECIALIZE(GDSGeometry)
PRECOMPILE_SPECIALIZE(GDSReader)
PRECOMPILE_SPECIALIZE(ToDiskMesh)
PRECOMPILE_SPECIALIZE(RateGrid)
PRECOMPILE_SPECIALIZE_PRECISION(Extrude)
PRECOMPILE_SPECIALIZE_PRECISION(Slice)

} // namespace viennaps
