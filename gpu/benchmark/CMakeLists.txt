project(ViennaPS-Benchmark)


add_gpu_executable(GPU_Benchmark target_name ${CMAKE_CURRENT_SOURCE_DIR}/GPU_Benchmark.cpp
                   ${CMAKE_SOURCE_DIR}/gpu/pipelines/SingleParticlePipeline.cu)

add_executable(CPU_Benchmark ${CMAKE_CURRENT_SOURCE_DIR}/CPU_Benchmark.cpp)
target_link_libraries(CPU_Benchmark ViennaPS)