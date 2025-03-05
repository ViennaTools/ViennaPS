project(ViennaPS-GPU)

option(VIENNAPS_ENABLE_OPTIXIR_SUPPORT
       "Enable support for generating OptiX-IR targeted input files" ON)

if(CUDA_VERSION VERSION_LESS 11.7)
  if(VIENNAPS_ENABLE_OPTIXIR_SUPPORT)
    message(
      SEND_ERROR "VIENNAPS_ENABLE_OPTIXIR_SUPPORT is not supported in CUDA versions less than 11.7")
  endif()
endif()

if(VIENNAPS_ENABLE_OPTIXIR_SUPPORT)
  option(VIENNAPS_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" ON)
  option(VIENNAPS_GENERATE_PTX "Generate PTX OptiX shaders" OFF)
else()
  option(VIENNAPS_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" OFF)
  option(VIENNAPS_GENERATE_PTX "Generate PTX OptiX shaders" ON)
endif()

#### Set variables
set(VIENNAPS_GPU_INCLUDE
    "${PROJECT_SOURCE_DIR}/include"
    CACHE STRING "Includes for ViennaPS GPU processing.")
set(VIENNAPS_PIPELINE_DIR
    "${PROJECT_SOURCE_DIR}/include/pipelines"
    CACHE STRING "Pre-built pipelines for OptiX")
set(VIENNAPS_CUDA_KERNELS
    ""
    CACHE STRING "ViennaPS CUDA kernel source files.")
include("../cmake/generate_ptx.cmake")

if(VIENNAPS_BUILD_EXAMPLES)
  message(STATUS "[ViennaPS] Adding GPU Examples")
  add_subdirectory(examples)
  add_subdirectory(benchmark)
endif(VIENNAPS_BUILD_EXAMPLES)

if(VIENNAPS_BUILD_TESTS)
  message(STATUS "[ViennaPS] Adding GPU Tests")
  add_subdirectory(tests)
endif()
