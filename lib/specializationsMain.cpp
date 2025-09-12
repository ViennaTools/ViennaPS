#include <process/psProcess.hpp>
#include <process/psProcessModel.hpp>
#include <psDomain.hpp>
#include <psDomainSetup.hpp>
#include <psExtrude.hpp>
#include <psGDSGeometry.hpp>
#include <psGDSReader.hpp>
#include <psPlanarize.hpp>
#include <psRateGrid.hpp>
#include <psReader.hpp>
#include <psToDiskMesh.hpp>
#include <psWriter.hpp>

#include <lsPreCompileMacros.hpp>

namespace viennaps {

// Precompile specializations for commonly used classes
PRECOMPILE_SPECIALIZE(Domain)
PRECOMPILE_SPECIALIZE(DomainSetup)
PRECOMPILE_SPECIALIZE(Process)
PRECOMPILE_SPECIALIZE(Planarize)
PRECOMPILE_SPECIALIZE(ProcessModelCPU)
PRECOMPILE_SPECIALIZE(Reader)
PRECOMPILE_SPECIALIZE(Writer)
PRECOMPILE_SPECIALIZE(GDSGeometry)
PRECOMPILE_SPECIALIZE(GDSReader)
PRECOMPILE_SPECIALIZE(ToDiskMesh)
PRECOMPILE_SPECIALIZE(RateGrid)
PRECOMPILE_SPECIALIZE_PRECISION(Extrude)

} // namespace viennaps