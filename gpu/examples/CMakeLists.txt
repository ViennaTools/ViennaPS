add_gpu_executable(holeGPU target_name ${CMAKE_CURRENT_SOURCE_DIR}/hole.cpp
                   ${CMAKE_SOURCE_DIR}/gpu/pipelines/SF6O2Pipeline.cu)

add_gpu_executable(waveguideGPU target_name ${CMAKE_CURRENT_SOURCE_DIR}/waveguide.cpp
                   ${CMAKE_SOURCE_DIR}/gpu/pipelines/FaradayCagePipeline.cu)

add_gpu_executable(trenchGPU target_name ${CMAKE_CURRENT_SOURCE_DIR}/trench.cpp
                   ${CMAKE_SOURCE_DIR}/gpu/pipelines/SingleParticlePipeline.cu)

add_gpu_executable(holeCompare target_name ${CMAKE_CURRENT_SOURCE_DIR}/holeCompare.cpp
                   ${CMAKE_SOURCE_DIR}/gpu/pipelines/IonBeamEtching.cu)
