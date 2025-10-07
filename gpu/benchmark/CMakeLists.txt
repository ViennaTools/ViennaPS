project(ViennaPS-GPU_Benchmark)

add_custom_target(ViennaPS-GPU_Benchmark ALL)

add_gpu_executable(GPU_Benchmark target_name ${CMAKE_CURRENT_SOURCE_DIR}/GPU_Benchmark.cpp
                   ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineTriangle.cu
                   ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineDisk.cu
                   ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineLine.cu
                   ${VIENNAPS_PIPELINE_DIR}/CallableWrapper.cu)
add_dependencies(ViennaPS-GPU_Benchmark GPU_Benchmark)

# add_gpu_executable(
#   GPU_Benchmark_single target_name ${CMAKE_CURRENT_SOURCE_DIR}/GPU_Benchmark_single.cpp
#   ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineTriangle.cu
#   ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineDisk.cu
#   ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineLine.cu)
# add_dependencies(ViennaPS-GPU_Benchmark GPU_Benchmark_single)

add_executable(CPU_Benchmark ${CMAKE_CURRENT_SOURCE_DIR}/CPU_Benchmark.cpp)
target_link_libraries(CPU_Benchmark ViennaPS)
add_dependencies(ViennaPS-GPU_Benchmark CPU_Benchmark)

add_executable(CPU_Benchmark_single ${CMAKE_CURRENT_SOURCE_DIR}/CPU_Benchmark_single.cpp)
target_link_libraries(CPU_Benchmark_single ViennaPS)
add_dependencies(ViennaPS-GPU_Benchmark CPU_Benchmark_single)

# add_gpu_executable(compare_flux target_name ${CMAKE_CURRENT_SOURCE_DIR}/compare_flux.cpp
#                    ${VIENNAPS_PIPELINE_DIR}/SingleParticlePipeline.cu)
# add_dependencies(ViennaPS-GPU_Benchmark compare_flux)
