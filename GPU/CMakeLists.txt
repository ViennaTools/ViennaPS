cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(ViennaPSGPUApplication)

find_package(ViennaPS CONFIG PATHS ${ViennaPS_BINARY_DIR})
if(NOT ViennaPS_FOUND)
  message(
    FATAL_ERROR
      "One of the dependencies hasn't been built yet and thus can't be used by the buildExamples target.\n\
  Try the following steps to fix the problem:\n\
  1. Turn off the buildExamples target by disabling the VIENNAPS_BUILD_EXAMPLES option when configuring the project\n\
  (e.g. with the `-D VIENNAPS_BUILD_EXAMPLES=OFF` flag)\n\
  2. Build the dependencies target: e.g. `make buildDependencies` when using Makefiles\n\
  3. Turn on the buildExamples target again by re-enabling the VIENNAPS_BUILD_EXAMPLES option\n\
  (e.g. with the `-D VIENNAPS_BUILD_EXAMPLES=ON` flag)\n\
  4. Now the configure step should pick up the built dependencies and proceed as intended.")
endif()

find_package(CGAL REQUIRED)

list(PREPEND VIENNAPS_INCLUDE_DIRS ${VIENNAPS_BUILD_INCLUDE_DIRS})

# Add subfolders to the include directories, since during installation the directory structure is
# flattened, but in the source tree - which we are using here - it is not.
list(PREPEND VIENNAPS_INCLUDE_DIRS ${VIENNAPS_BUILD_INCLUDE_DIRS}/CellSet
     ${VIENNAPS_BUILD_INCLUDE_DIRS}/Geometries ${VIENNAPS_BUILD_INCLUDE_DIRS}/Models
     ${VIENNAPS_BUILD_INCLUDE_DIRS}/Compact)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL TRUE)

if("${is_cached}" STREQUAL "" AND DEFINED OptiX_INSTALL_DIR)
  # Required for windows to convert backslashes to forward slashes
  file(TO_CMAKE_PATH "${OptiX_INSTALL_DIR}" OptiX_INSTALL_DIR)
  set(OptiX_INSTALL_DIR
      "${OptiX_INSTALL_DIR}"
      CACHE PATH "Path to OptiX installation" FORCE)
else()
  set(OptiX_INSTALL_DIR
      $ENV{OptiX_INSTALL_DIR}
      CACHE PATH "Path to OptiX installation.")
endif()
include(cmake/configure_optix.cmake)

set(VIENNAPS_GPU_INCLUDE_DIRS #
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/geometry ${PROJECT_SOURCE_DIR}/process
    ${PROJECT_SOURCE_DIR}/rayTracing ${PROJECT_SOURCE_DIR}/utils ${PROJECT_SOURCE_DIR}/app)

add_custom_target(buildGPUApplication)

include_directories(${VIENNAPS_GPU_INCLUDE_DIRS} )
file(GLOB kernels "${PROJECT_SOURCE_DIR}/deviceKernels/*.cu")
foreach(kernel ${kernels})
  cuda_compile_ptx(
    cuda_ptx_files
    ${kernel}
    OPTIONS
    --generate-line-info
    -use_fast_math
    --keep
    -gencode=arch=compute_80,code=sm_80
    --expt-relaxed-constexpr)
  get_filename_component(kernelName ${kernel} NAME_WE)

  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/${kernelName}.ptx
    COMMAND ${CMAKE_COMMAND} -E rename ${cuda_ptx_files} ${CMAKE_BINARY_DIR}/${kernelName}.ptx
    DEPENDS ${cuda_ptx_files})

  add_custom_target(
    ${kernelName} ALL
    DEPENDS ${CMAKE_BINARY_DIR}/${kernelName}.ptx ${kernel}
    SOURCES ${kernel})

  add_dependencies(buildGPUApplication ${kernelName})
endforeach()

cuda_compile_and_embed(embedded_SF6O2_pipeline ${PROJECT_SOURCE_DIR}/pipelines/SF6O2rayPipeline.cu)
cuda_compile_and_embed(embedded_Fluorocarbon_pipeline
                       ${PROJECT_SOURCE_DIR}/pipelines/FluorocarbonPipeline.cu)

cuda_compile_and_embed(embedded_deposition_pipeline
                       ${PROJECT_SOURCE_DIR}/pipelines/DepositionPipeline.cu)
add_compile_definitions(GPU_SUPPORT)

add_executable(
  ViennaPS_GPU ${embedded_SF6O2_pipeline} ${embedded_deposition_pipeline}
               ${embedded_Fluorocarbon_pipeline} ${PROJECT_SOURCE_DIR}/app/pscuApplication.cpp)

target_include_directories(ViennaPS_GPU PUBLIC ${VIENNAPS_INCLUDE_DIRS} ${OptiX_INCLUDE}
                                               ${VIENNAPS_GPU_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/app)

target_link_libraries(ViennaPS_GPU CGAL::CGAL ${Boost_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${VIENNAPS_LIBRARIES})

add_dependencies(buildGPUApplication ViennaPS_GPU)

add_executable(
  ViennaPS_GPU_Benchmark  ${embedded_deposition_pipeline}
                ${PROJECT_SOURCE_DIR}/benchmark/GPU_Benchmark.cpp)

target_include_directories(ViennaPS_GPU_Benchmark PUBLIC ${VIENNAPS_INCLUDE_DIRS} ${OptiX_INCLUDE}
                                               ${VIENNAPS_GPU_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/app)

target_link_libraries(ViennaPS_GPU_Benchmark ${Boost_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${VIENNAPS_LIBRARIES})

# add_dependencies(buildGPUBenchmark ViennaPS_GPU_Benchmark)
