#include <gds/psGDSGeometry.hpp>
#include <gds/psGDSReader.hpp>

#include <process/psProcess.hpp>
#include <process/psProcessModel.hpp>

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

namespace viennaps {

// Precompile specializations for commonly used classes
PRECOMPILE_SPECIALIZE(Domain)
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
PRECOMPILE_SPECIALIZE_PRECISION(Slice)

} // namespace viennaps

// ----------------------------------------------------------------
// Specializations for process models
// ----------------------------------------------------------------

#include <models/psCF4O2Etching.hpp>
#include <models/psCSVFileProcess.hpp>
#include <models/psDirectionalProcess.hpp>
#include <models/psFaradayCageEtching.hpp>
#include <models/psFluorocarbonEtching.hpp>
#include <models/psGeometricDistributionModels.hpp>
#include <models/psHBrO2Etching.hpp>
#include <models/psIonBeamEtching.hpp>
#include <models/psIsotropicProcess.hpp>
#include <models/psMultiParticleProcess.hpp>
#include <models/psOxideRegrowth.hpp>
#include <models/psSF6C4F8Etching.hpp>
#include <models/psSF6O2Etching.hpp>
#include <models/psSelectiveEpitaxy.hpp>
#include <models/psSingleParticleALD.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <models/psTEOSDeposition.hpp>
#include <models/psTEOSPECVD.hpp>
#include <models/psWetEtching.hpp>

namespace viennaps {

// Precompile specializations for process models
// Emulation Models
PRECOMPILE_SPECIALIZE(CSVFileProcess)
PRECOMPILE_SPECIALIZE(DirectionalProcess)
PRECOMPILE_SPECIALIZE(IsotropicProcess)

// Geometric Models
PRECOMPILE_SPECIALIZE(BoxDistribution)
PRECOMPILE_SPECIALIZE(SphereDistribution)

// Plasma Etching Models
PRECOMPILE_SPECIALIZE(CF4O2Etching)
PRECOMPILE_SPECIALIZE(HBrO2Etching)
PRECOMPILE_SPECIALIZE(SF6O2Etching)
PRECOMPILE_SPECIALIZE(SF6C4F8Etching)
PRECOMPILE_SPECIALIZE(FluorocarbonEtching)

// Flux Based Models
PRECOMPILE_SPECIALIZE(SingleParticleProcess)
PRECOMPILE_SPECIALIZE(MultiParticleProcess)

// TEOS Models
PRECOMPILE_SPECIALIZE(TEOSDeposition)
PRECOMPILE_SPECIALIZE(TEOSPECVD)

// Ion Beam Etching Models
PRECOMPILE_SPECIALIZE(FaradayCageEtching)
PRECOMPILE_SPECIALIZE(IonBeamEtching)

// Crystal Anisotropy Models
PRECOMPILE_SPECIALIZE(SelectiveEpitaxy)
PRECOMPILE_SPECIALIZE(WetEtching)

// Atomic Layer Processing Models
PRECOMPILE_SPECIALIZE(SingleParticleALD)

// Other Models
PRECOMPILE_SPECIALIZE(OxideRegrowth)

} // namespace viennaps

// ----------------------------------------------------------------
// Specializations for geometries
// ----------------------------------------------------------------

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