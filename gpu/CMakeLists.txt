project(ViennaPSGPUApplication)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL TRUE)

# if("${is_cached}" STREQUAL "" AND DEFINED OptiX_INSTALL_DIR)
#   # Required for windows to convert backslashes to forward slashes
#   file(TO_CMAKE_PATH "${OptiX_INSTALL_DIR}" OptiX_INSTALL_DIR)
#   set(OptiX_INSTALL_DIR
#       "${OptiX_INSTALL_DIR}"
#       CACHE PATH "Path to OptiX installation" FORCE)
# else()
#   set(OptiX_INSTALL_DIR
#       $ENV{OptiX_INSTALL_DIR}
#       CACHE PATH "Path to OptiX installation.")
# endif()
set(OptiX_INSTALL_DIR "/home/tobias/OptiX" CACHE PATH "Path to OptiX installed location.")
set(OptiX_INCLUDE "${OptiX_INSTALL_DIR}/include")
message(STATUS "OptiX_INSTALL_DIR: ${OptiX_INSTALL_DIR}")
message(STATUS "OptiX_INCLUDE: ${OptiX_INCLUDE}")
include(${CMAKE_SOURCE_DIR}/cmake/configure_optix.cmake)

set(VIENNAPS_GPU_INCLUDE_DIRS #
    ${CMAKE_SOURCE_DIR}/app
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/geometry ${PROJECT_SOURCE_DIR}/process
    ${PROJECT_SOURCE_DIR}/rayTracing ${PROJECT_SOURCE_DIR}/utils ${PROJECT_SOURCE_DIR}/app)

add_custom_target(buildGPUApplication)

include_directories(${VIENNAPS_GPU_INCLUDE_DIRS})
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

# cuda_compile_and_embed(embedded_SF6O2_pipeline ${PROJECT_SOURCE_DIR}/pipelines/SF6O2Pipeline.cu)
# cuda_compile_and_embed(embedded_Fluorocarbon_pipeline
#                        ${PROJECT_SOURCE_DIR}/pipelines/FluorocarbonPipeline.cu)

cuda_compile_and_embed(embedded_deposition_pipeline
                       ${PROJECT_SOURCE_DIR}/pipelines/DepositionPipeline.cu)
add_compile_definitions(GPU_SUPPORT)

add_executable(
  ViennaPS_GPU ${embedded_SF6O2_pipeline} ${embedded_deposition_pipeline}
               ${embedded_Fluorocarbon_pipeline} ${PROJECT_SOURCE_DIR}/app/gpuApplication.cpp)
target_include_directories(ViennaPS_GPU PUBLIC ${OptiX_INCLUDE} ${VIENNAPS_GPU_INCLUDE_DIRS})
target_link_libraries(ViennaPS_GPU  ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ViennaPS)

add_dependencies(buildGPUApplication ViennaPS_GPU)

### BENCHMARK ###

# add_executable(ViennaPS_GPU_Benchmark ${embedded_deposition_pipeline}
#                                       ${PROJECT_SOURCE_DIR}/benchmark/GPU_Benchmark.cpp)

# target_include_directories(
#   ViennaPS_GPU_Benchmark PUBLIC ${VIENNAPS_INCLUDE_DIRS} ${OptiX_INCLUDE}
#                                 ${VIENNAPS_GPU_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/app)
# target_link_libraries(ViennaPS_GPU_Benchmark ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ViennaPS)

### EXAMPLES ###

# add_custom_target(buildGPUExamples)

# add_executable(GPU_Trench ${embedded_SF6O2_pipeline}
#                                       ${PROJECT_SOURCE_DIR}/examples/trench.cpp)
# target_include_directories(
#   GPU_Trench PUBLIC ${VIENNAPS_INCLUDE_DIRS} ${OptiX_INCLUDE}
#                                 ${VIENNAPS_GPU_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/app)
# target_link_libraries(GPU_Trench ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ViennaPS)
# add_dependencies(buildGPUExamples GPU_Trench)

# add_executable(GPU_Hole ${embedded_SF6O2_pipeline}
#                                       ${PROJECT_SOURCE_DIR}/examples/hole.cpp)
# target_include_directories(
#   GPU_Hole PUBLIC ${VIENNAPS_INCLUDE_DIRS} ${OptiX_INCLUDE}
#                                 ${VIENNAPS_GPU_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/app)
# target_link_libraries(GPU_Hole ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${VIENNAPS_LIBRARIES})
# add_dependencies(buildGPUExamples GPU_Hole)
