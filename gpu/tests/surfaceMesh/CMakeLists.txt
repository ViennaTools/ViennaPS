project(surfaceMesh LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)
target_include_directories(${PROJECT_NAME} PRIVATE ${VIENNAPS_GPU_INCLUDE} ${VIENNARAY_GPU_INCLUDE})

add_dependencies(ViennaPS_GPU_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
