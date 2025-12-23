#### Set variables
set(VIENNAPS_GPU_MODELS
    "${CMAKE_SOURCE_DIR}/gpu/models"
    CACHE STRING "ViennaPS GPU process models.")

# Incldue directories for PTX and optix-ir compilation (force cache to update with new entries)
set(VIENNACORE_PTX_INCLUDE_DIRS
    ${VIENNACORE_PTX_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/include/viennaps"
    CACHE STRING "PTX include directories" FORCE)

# Add the callable wrapper
viennacore_add_optixir(ViennaPSCallableWrapper ${VIENNAPS_GPU_MODELS}/CallableWrapper.cu)

set(VIENNAPS_GPU_DEPENDENCIES
    ViennaPSCallableWrapper ${VIENNARAY_GPU_DEPENDENCIES}
    CACHE STRING "GPU-related dependencies for ViennaPS targets")

if(VIENNAPS_BUILD_EXAMPLES)
  message(STATUS "[ViennaPS] Adding GPU Benchmarks")
  add_subdirectory(benchmark)
endif(VIENNAPS_BUILD_EXAMPLES)
