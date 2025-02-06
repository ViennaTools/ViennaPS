project(testReflections LANGUAGES CXX)

# test kernel
cuda_wrap_srcs(
    ${PROJECT_NAME} # target name
    PTX
    generated_files
    "testReflections.cu")
    #${cmake_options}
    #OPTIONS
    #${options})

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp" ${generated_files})
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
target_compile_definitions(${PROJECT_NAME} PRIVATE
    VIENNACORE_COMPILE_GPU
    VIENNAPS_KERNELS_PATH_DEFINE=${VIENNAPS_PTX_DIR}
    PSCU_TEST)

add_dependencies(ViennaPS-GPU_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
