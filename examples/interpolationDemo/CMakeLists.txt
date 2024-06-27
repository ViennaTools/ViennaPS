project(interpolationDemo LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)

configure_file(griddata.csv ${CMAKE_CURRENT_BINARY_DIR}/griddata.csv COPYONLY)
configure_file(scatterdata.csv ${CMAKE_CURRENT_BINARY_DIR}/scatterdata.csv COPYONLY)
configure_file(plot.py ${CMAKE_CURRENT_BINARY_DIR}/plot.py COPYONLY)

add_dependencies(ViennaPS_Examples ${PROJECT_NAME})
viennacore_setup_bat(${PROJECT_NAME} ${VIENNAPS_EXAMPLES_LIB})
