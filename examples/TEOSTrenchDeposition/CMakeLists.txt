project(TEOSTrenchDeposition LANGUAGES CXX)

add_executable(singleTEOS "singleTEOS.cpp")
target_link_libraries(singleTEOS PRIVATE ViennaPS)

configure_file(singleTEOS_config.txt ${CMAKE_CURRENT_BINARY_DIR}/singleTEOS_config.txt COPYONLY)

add_dependencies(ViennaPS_Examples singleTEOS)
viennacore_setup_bat(singleTEOS ${VIENNAPS_ARTIFACTS_DIRECTORY})

add_executable(multiTEOS "multiTEOS.cpp")
target_link_libraries(multiTEOS PRIVATE ViennaPS)

configure_file(multiTEOS_config.txt ${CMAKE_CURRENT_BINARY_DIR}/multiTEOS_config.txt COPYONLY)

add_dependencies(ViennaPS_Examples multiTEOS)
viennacore_setup_bat(multiTEOS ${VIENNAPS_ARTIFACTS_DIRECTORY})
