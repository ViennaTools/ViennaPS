project(ViennaPS-GPU_Benchmark)

add_custom_target(ViennaPS-GPU_Benchmark ALL)

add_gpu_executable(GPU_Benchmark target_name ${CMAKE_CURRENT_SOURCE_DIR}/GPU_Benchmark.cpp)
add_dependencies(ViennaPS-GPU_Benchmark GPU_Benchmark)

add_executable(CPU_Benchmark ${CMAKE_CURRENT_SOURCE_DIR}/CPU_Benchmark.cpp)
target_link_libraries(CPU_Benchmark ViennaPS)
add_dependencies(ViennaPS-GPU_Benchmark CPU_Benchmark)

add_executable(CPU_Benchmark_single ${CMAKE_CURRENT_SOURCE_DIR}/CPU_Benchmark_single.cpp)
target_link_libraries(CPU_Benchmark_single ViennaPS)
add_dependencies(ViennaPS-GPU_Benchmark CPU_Benchmark_single)

add_gpu_executable(holeComparison target_name ${CMAKE_CURRENT_SOURCE_DIR}/holeCompare.cpp)
add_dependencies(ViennaPS-GPU_Benchmark holeComparison)
