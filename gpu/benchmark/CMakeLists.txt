project(ViennaPS-GPU_Benchmark)

add_custom_target(ViennaPS-GPU_Benchmark ALL)

add_executable(GPU_Benchmark ${CMAKE_CURRENT_SOURCE_DIR}/GPU_Benchmark.cpp)
target_link_libraries(GPU_Benchmark PRIVATE ViennaPS)
add_dependencies(GPU_Benchmark ${VIENNAPS_GPU_DEPENDENCIES})
add_dependencies(ViennaPS-GPU_Benchmark GPU_Benchmark)

add_executable(CPU_Benchmark ${CMAKE_CURRENT_SOURCE_DIR}/CPU_Benchmark.cpp)
target_link_libraries(CPU_Benchmark PRIVATE ViennaPS)
add_dependencies(CPU_Benchmark ${VIENNAPS_GPU_DEPENDENCIES})
add_dependencies(ViennaPS-GPU_Benchmark CPU_Benchmark)

add_executable(holeComparison ${CMAKE_CURRENT_SOURCE_DIR}/holeCompare.cpp)
target_link_libraries(holeComparison PRIVATE ViennaPS)
add_dependencies(holeComparison ${VIENNAPS_GPU_DEPENDENCIES})
add_dependencies(ViennaPS-GPU_Benchmark holeComparison)

add_executable(holeComparisonGrid ${CMAKE_CURRENT_SOURCE_DIR}/holeCompareGrid.cpp)
target_link_libraries(holeComparisonGrid PRIVATE ViennaPS)
add_dependencies(holeComparisonGrid ${VIENNAPS_GPU_DEPENDENCIES})
add_dependencies(ViennaPS-GPU_Benchmark holeComparisonGrid)
