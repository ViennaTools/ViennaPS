project(TEOSTrenchDeposition LANGUAGES CXX)

add_executable(singleTEOS "singleTEOS.cpp")
target_link_libraries(singleTEOS PRIVATE ViennaPS)

configure_file(singleTEOS.py ${CMAKE_CURRENT_BINARY_DIR}/singleTEOS.py COPYONLY)
configure_file(singleTEOS_config.txt ${CMAKE_CURRENT_BINARY_DIR}/singleTEOS_config.txt COPYONLY)

add_dependencies(ViennaPS_Examples singleTEOS)
viennacore_setup_bat(singleTEOS ${CMAKE_CURRENT_BINARY_DIR})

add_executable(multiTEOS "multiTEOS.cpp")
target_link_libraries(multiTEOS PRIVATE ViennaPS)

configure_file(multiTEOS.py ${CMAKE_CURRENT_BINARY_DIR}/multiTEOS.py COPYONLY)
configure_file(multiTEOS_config.txt ${CMAKE_CURRENT_BINARY_DIR}/multiTEOS_config.txt COPYONLY)

add_dependencies(ViennaPS_Examples multiTEOS)
viennacore_setup_bat(multiTEOS ${CMAKE_CURRENT_BINARY_DIR})
