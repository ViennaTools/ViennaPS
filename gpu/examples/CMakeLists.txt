add_GPU_executable( custom target_name
                    ${CMAKE_CURRENT_SOURCE_DIR}/custom.cpp
                    ${CMAKE_SOURCE_DIR}/gpu/pipelines/SingleParticlePipeline.cu
                    )