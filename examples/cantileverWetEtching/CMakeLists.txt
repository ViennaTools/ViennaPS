project(cantileverWetEtching LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)

configure_file(cantileverWetEtching.py ${CMAKE_CURRENT_BINARY_DIR}/cantileverWetEtching.py COPYONLY)
configure_file(cantilever_mask.gds ${CMAKE_CURRENT_BINARY_DIR}/cantilever_mask.gds COPYONLY)

add_dependencies(ViennaPS_Examples ${PROJECT_NAME})
setup_windows_bat(${PROJECT_NAME} ${VIENNAPS_EXAMPLES_LIB})
