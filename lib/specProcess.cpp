#include <process/psProcess.hpp>
#include <process/psProcessModel.hpp>

#include <lsPreCompileMacros.hpp>

namespace viennaps {

// Precompile specializations for commonly used classes
PRECOMPILE_SPECIALIZE(Process)
PRECOMPILE_SPECIALIZE(ProcessModelCPU)

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {
PRECOMPILE_SPECIALIZE(ProcessModelGPU)
} // namespace gpu
#endif

} // namespace viennaps
