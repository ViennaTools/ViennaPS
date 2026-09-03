project(voxelNoise LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)
target_compile_definitions(
  ${PROJECT_NAME}
  PRIVATE
    VIENNAPS_MECHANISM_DIR="${CMAKE_CURRENT_LIST_DIR}/../surfaceChemistry/reactions")

add_dependencies(ViennaPS_Examples ${PROJECT_NAME})
