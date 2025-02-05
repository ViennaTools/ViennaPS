include_directories(${CMAKE_SOURCE_DIR}/include/viennaps/models)

add_gpu_executable(hole target_name ${CMAKE_CURRENT_SOURCE_DIR}/hole.cpp
                   ${CMAKE_SOURCE_DIR}/gpu/pipelines/SF6O2Pipeline.cu)

add_gpu_executable(waveguide target_name ${CMAKE_CURRENT_SOURCE_DIR}/waveguide.cpp
                   ${CMAKE_SOURCE_DIR}/gpu/pipelines/FaradayCagePipeline.cu)

add_gpu_executable(trench target_name ${CMAKE_CURRENT_SOURCE_DIR}/trench.cpp
                   ${CMAKE_SOURCE_DIR}/gpu/pipelines/SingleParticlePipeline.cu)