project(voxelGpuFlux LANGUAGES CXX)

if(VIENNAPS_USE_GPU)
  add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
  target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)
  target_compile_definitions(
    ${PROJECT_NAME}
    PRIVATE
      VIENNAPS_MECHANISM_DIR="${CMAKE_CURRENT_LIST_DIR}/../../examples/surfaceChemistry/reactions")

  add_dependencies(ViennaPS_Tests ${PROJECT_NAME})
  add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
endif()
