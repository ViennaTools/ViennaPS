#include <models/psCSVFileProcess.hpp>
#include <models/psDirectionalProcess.hpp>
#include <models/psGeometricDistributionModels.hpp>
#include <models/psIsotropicProcess.hpp>
#include <models/psSelectiveEpitaxy.hpp>
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

// Crystal Anisotropy Models
PRECOMPILE_SPECIALIZE(SelectiveEpitaxy)
PRECOMPILE_SPECIALIZE(WetEtching)

} // namespace viennaps