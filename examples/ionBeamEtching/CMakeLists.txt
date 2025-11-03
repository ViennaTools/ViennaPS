project(ionBeamEtching LANGUAGES CXX)

if(VIENNAPS_USE_GPU)
  add_gpu_executable(${PROJECT_NAME} target_name "${PROJECT_NAME}.cpp")
else()
  add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
  target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)
endif()

configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)

add_dependencies(ViennaPS_Examples ${PROJECT_NAME})
viennacore_setup_bat(${PROJECT_NAME} ${VIENNAPS_ARTIFACTS_DIRECTORY})
