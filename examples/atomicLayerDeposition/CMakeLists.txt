project(atomicLayerDeposition LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)

configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)
configure_file(atomicLayerDeposition.py ${CMAKE_CURRENT_BINARY_DIR}/atomicLayerDeposition.py
               COPYONLY)

add_dependencies(ViennaPS_Examples ${PROJECT_NAME})
setup_windows_bat(${PROJECT_NAME} ${VIENNAPS_EXAMPLES_LIB})
