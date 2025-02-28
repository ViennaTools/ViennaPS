project(ViennaPS-GPU_Benchmark)
include("../../cmake/generate_ptx.cmake")

add_gpu_executable(GPU_Benchmark target_name ${CMAKE_CURRENT_SOURCE_DIR}/GPU_Benchmark.cpp
                   ${CMAKE_SOURCE_DIR}/gpu/pipelines/SingleParticlePipeline.cu)

add_gpu_executable(
  GPU_Benchmark_single target_name ${CMAKE_CURRENT_SOURCE_DIR}/GPU_Benchmark_single.cpp
  ${CMAKE_SOURCE_DIR}/gpu/pipelines/SingleParticlePipeline.cu)

add_executable(CPU_Benchmark ${CMAKE_CURRENT_SOURCE_DIR}/CPU_Benchmark.cpp)
target_link_libraries(CPU_Benchmark ViennaPS)

add_executable(CPU_Benchmark_single ${CMAKE_CURRENT_SOURCE_DIR}/CPU_Benchmark_single.cpp)
target_link_libraries(CPU_Benchmark_single ViennaPS)

add_gpu_executable(compare_flux target_name ${CMAKE_CURRENT_SOURCE_DIR}/compare_flux.cpp
                   ${CMAKE_SOURCE_DIR}/gpu/pipelines/SingleParticlePipeline.cu)
