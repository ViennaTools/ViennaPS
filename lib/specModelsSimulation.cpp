#include <models/psCF4O2Etching.hpp>
#include <models/psFaradayCageEtching.hpp>
#include <models/psFluorocarbonEtching.hpp>
#include <models/psHBrO2Etching.hpp>
#include <models/psIonBeamEtching.hpp>
#include <models/psMultiParticleProcess.hpp>
#include <models/psOxideRegrowth.hpp>
#include <models/psSF6C4F8Etching.hpp>
#include <models/psSF6O2Etching.hpp>
#include <models/psSingleParticleALD.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <models/psTEOSDeposition.hpp>
#include <models/psTEOSPECVD.hpp>

namespace viennaps {

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

// Atomic Layer Processing Models
PRECOMPILE_SPECIALIZE(SingleParticleALD)

// Other Models
PRECOMPILE_SPECIALIZE(OxideRegrowth)

} // namespace viennaps