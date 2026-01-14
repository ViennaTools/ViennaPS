find_package(FiDiTi 0.0.0 CONFIG)

if(NOT TARGET FiDiTi::FiDiTi)
  message(STATUS "[ViennaPS] FDTD library not found, skipping example")
  return()
endif()

project(photodiode LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS FiDiTi::FiDiTi)

configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)

add_dependencies(ViennaPS_Examples ${PROJECT_NAME})
viennacore_setup_bat(${PROJECT_NAME} ${VIENNAPS_ARTIFACTS_DIRECTORY})
