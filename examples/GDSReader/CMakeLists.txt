project(GDSReader LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)

add_executable(GDSReader2D "GDSReader2D.cpp")
target_link_libraries(GDSReader2D PRIVATE ViennaPS)

configure_file(mask.gds ${CMAKE_CURRENT_BINARY_DIR}/mask.gds COPYONLY)
configure_file(myTest.gds ${CMAKE_CURRENT_BINARY_DIR}/myTest.gds COPYONLY)
configure_file(test_structure.txt ${CMAKE_CURRENT_BINARY_DIR}/test_structure.txt COPYONLY)
configure_file(plotCSV.py ${CMAKE_CURRENT_BINARY_DIR}/plotCSV.py COPYONLY)

add_dependencies(ViennaPS_Examples ${PROJECT_NAME})
viennacore_setup_bat(${PROJECT_NAME} ${VIENNAPS_ARTIFACTS_DIRECTORY})

add_dependencies(ViennaPS_Examples GDSReader2D)
viennacore_setup_bat(GDSReader2D ${VIENNAPS_ARTIFACTS_DIRECTORY})
