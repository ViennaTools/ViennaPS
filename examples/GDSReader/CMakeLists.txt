project(GDSReader LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)

configure_file(GDSReader.py ${CMAKE_CURRENT_BINARY_DIR}/GDSReader.py COPYONLY)
configure_file(mask.gds ${CMAKE_CURRENT_BINARY_DIR}/mask.gds COPYONLY)

add_dependencies(ViennaPS_Examples ${PROJECT_NAME})
viennacore_setup_bat(${PROJECT_NAME} ${VIENNAPS_EXAMPLES_LIB})
