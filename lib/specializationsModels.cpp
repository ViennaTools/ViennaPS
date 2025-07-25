#include <models/psAnisotropicProcess.hpp>
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
#include <models/psSingleParticleALD.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <models/psTEOSDeposition.hpp>
#include <models/psTEOSPECVD.hpp>

namespace viennaps {

// Precompile specializations for process models
PRECOMPILE_SPECIALIZE(AnisotropicProcess)
PRECOMPILE_SPECIALIZE(CF4O2Etching)
PRECOMPILE_SPECIALIZE(CSVFileProcess)
PRECOMPILE_SPECIALIZE(DirectionalProcess)
PRECOMPILE_SPECIALIZE(FaradayCageEtching)
PRECOMPILE_SPECIALIZE(FluorocarbonEtching)
PRECOMPILE_SPECIALIZE(HBrO2Etching)
PRECOMPILE_SPECIALIZE(IonBeamEtching)
PRECOMPILE_SPECIALIZE(IsotropicProcess)
PRECOMPILE_SPECIALIZE(MultiParticleProcess)
PRECOMPILE_SPECIALIZE(OxideRegrowth)
PRECOMPILE_SPECIALIZE(SF6C4F8Etching)
PRECOMPILE_SPECIALIZE(SF6O2Etching)
PRECOMPILE_SPECIALIZE(SingleParticleALD)
PRECOMPILE_SPECIALIZE(SingleParticleProcess)
PRECOMPILE_SPECIALIZE(TEOSDeposition)
PRECOMPILE_SPECIALIZE(TEOSPECVD)
PRECOMPILE_SPECIALIZE(SphereDistribution)
PRECOMPILE_SPECIALIZE(BoxDistribution)

} // namespace viennaps