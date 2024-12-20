add_gpu_executable(custom target_name ${CMAKE_CURRENT_SOURCE_DIR}/custom.cpp
                   ${CMAKE_SOURCE_DIR}/gpu/pipelines/SingleParticlePipeline.cu)

include_directories(${CMAKE_SOURCE_DIR}/include/viennaps/models)
add_gpu_executable(hole target_name ${CMAKE_CURRENT_SOURCE_DIR}/hole.cpp
                   ${CMAKE_SOURCE_DIR}/gpu/pipelines/SF6O2Pipeline.cu)
