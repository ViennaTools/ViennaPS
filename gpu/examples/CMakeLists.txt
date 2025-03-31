project(ViennaPS-GPU_Examples)

add_custom_target(ViennaPS-GPU_Examples ALL)

add_gpu_executable(holeGPU target_name ${CMAKE_CURRENT_SOURCE_DIR}/hole.cpp
                   ${VIENNAPS_PIPELINE_DIR}/SF6O2Pipeline.cu)
add_dependencies(ViennaPS-GPU_Examples holeGPU)

add_gpu_executable(waveguideGPU target_name ${CMAKE_CURRENT_SOURCE_DIR}/waveguide.cpp
                   ${VIENNAPS_PIPELINE_DIR}/FaradayCagePipeline.cu)
add_dependencies(ViennaPS-GPU_Examples waveguideGPU)

add_gpu_executable(trenchGPU target_name ${CMAKE_CURRENT_SOURCE_DIR}/trench.cpp
                   ${VIENNAPS_PIPELINE_DIR}/MultiParticlePipeline.cu)
add_dependencies(ViennaPS-GPU_Examples trenchGPU)

add_gpu_executable(holeCompare target_name ${CMAKE_CURRENT_SOURCE_DIR}/holeCompare.cpp
                   ${VIENNAPS_PIPELINE_DIR}/IonBeamPipeline.cu)
add_dependencies(ViennaPS-GPU_Examples holeCompare)