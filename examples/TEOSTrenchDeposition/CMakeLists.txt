project(TEOSTrenchDeposition LANGUAGES CXX)

add_executable(singleTEOS "singleTEOS.cpp")
target_link_libraries(singleTEOS PRIVATE ViennaPS)

configure_file(singleTEOS.py ${CMAKE_CURRENT_BINARY_DIR}/singleTEOS.py COPYONLY)
configure_file(singleTEOS_config.txt ${CMAKE_CURRENT_BINARY_DIR}/singleTEOS_config.txt COPYONLY)

add_dependencies(ViennaPS_Examples singleTEOS)
setup_windows_bat(singleTEOS ${VIENNAPS_EXAMPLES_LIB})

add_executable(multiTEOS "multiTEOS.cpp")
target_link_libraries(multiTEOS PRIVATE ViennaPS)

configure_file(multiTEOS.py ${CMAKE_CURRENT_BINARY_DIR}/multiTEOS.py COPYONLY)
configure_file(multiTEOS_config.txt ${CMAKE_CURRENT_BINARY_DIR}/multiTEOS_config.txt COPYONLY)

add_dependencies(ViennaPS_Examples multiTEOS)
setup_windows_bat(multiTEOS ${VIENNAPS_EXAMPLES_LIB})
