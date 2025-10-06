project(ViennaPS-GPU_Examples)

add_custom_target(ViennaPS-GPU_Examples ALL)

# add_gpu_executable(waveguideGPU target_name ${CMAKE_CURRENT_SOURCE_DIR}/waveguide.cpp
#                    ${VIENNAPS_PIPELINE_DIR}/FaradayCagePipeline.cu)
# add_dependencies(ViennaPS-GPU_Examples waveguideGPU)

add_gpu_executable(holeCompare target_name ${CMAKE_CURRENT_SOURCE_DIR}/holeCompare.cpp
                   ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineDisk.cu
                   ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineTriangle.cu
                   ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineLine.cu
                   ${VIENNAPS_PIPELINE_DIR}/CallableWrapper.cu)
add_dependencies(ViennaPS-GPU_Examples holeCompare)
