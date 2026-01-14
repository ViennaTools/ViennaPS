#### Set variables
set(VIENNAPS_GPU_MODELS
    "${PROJECT_SOURCE_DIR}/gpu/models"
    CACHE STRING "ViennaPS GPU process models.")

# Incldue directories for PTX and optix-ir compilation (force cache to update with new entries)
if(NOT VIENNAPS_PASSED_FIRST_CONFIGURE)
  set(VIENNACORE_NVCC_INCLUDE_DIRS
      ${VIENNACORE_NVCC_INCLUDE_DIRS} "${PROJECT_SOURCE_DIR}/include/viennaps"
      CACHE STRING "PTX include directories" FORCE)
endif()

# Add the callable wrapper
viennacore_add_optixir(
  ViennaPSCallableWrapper
  ${VIENNAPS_GPU_MODELS}/CallableWrapper.cu
  DEPENDS
  "${VIENNAPS_GPU_MODELS}/FaradayCage.cuh"
  "${VIENNAPS_GPU_MODELS}/IonBeamEtching.cuh"
  "${VIENNAPS_GPU_MODELS}/MultiParticle.cuh"
  "${VIENNAPS_GPU_MODELS}/PlasmaEtching.cuh"
  "${VIENNAPS_GPU_MODELS}/SingleParticle.cuh"
  "${VIENNAPS_GPU_MODELS}/SingleParticleALD.cuh"
  "${VIENNAPS_GPU_MODELS}/TEOSPECVD.cuh")

set(VIENNAPS_GPU_DEPENDENCIES
    ViennaPSCallableWrapper ${VIENNARAY_GPU_DEPENDENCIES}
    CACHE STRING "GPU-related dependencies for ViennaPS targets")

if(VIENNAPS_BUILD_EXAMPLES)
  message(STATUS "[ViennaPS] Adding GPU Benchmarks")
  add_subdirectory(benchmark)
endif(VIENNAPS_BUILD_EXAMPLES)

set(VIENNAPS_PASSED_FIRST_CONFIGURE
    ON
    CACHE INTERNAL "")
