project(ViennaPS-GPU_Examples)

add_custom_target(ViennaPS-GPU_Examples ALL)

add_gpu_executable(waveguideGPU target_name ${CMAKE_CURRENT_SOURCE_DIR}/waveguide.cpp)
add_dependencies(ViennaPS-GPU_Examples waveguideGPU)

add_gpu_executable(holeCompare target_name ${CMAKE_CURRENT_SOURCE_DIR}/holeCompare.cpp)
add_dependencies(ViennaPS-GPU_Examples holeCompare)
