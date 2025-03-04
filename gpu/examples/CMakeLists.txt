project(ViennaPS-GPU_Examples)

add_gpu_executable(holeGPU target_name ${CMAKE_CURRENT_SOURCE_DIR}/hole.cpp
                   ${VIENNAPS_PIPELINE_DIR}/SF6O2Pipeline.cu)

add_gpu_executable(waveguideGPU target_name ${CMAKE_CURRENT_SOURCE_DIR}/waveguide.cpp
                   ${VIENNAPS_PIPELINE_DIR}/FaradayCagePipeline.cu)

add_gpu_executable(trenchGPU target_name ${CMAKE_CURRENT_SOURCE_DIR}/trench.cpp
                   ${VIENNAPS_PIPELINE_DIR}/MultiParticlePipeline.cu)

add_gpu_executable(holeCompare target_name ${CMAKE_CURRENT_SOURCE_DIR}/holeCompare.cpp
                   ${VIENNAPS_PIPELINE_DIR}/IonBeamPipeline.cu)
