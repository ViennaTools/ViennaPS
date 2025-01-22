project(holeEtching LANGUAGES CXX)

add_executable(${PROJECT_NAME}.exe "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME}.exe PRIVATE ViennaPS)

configure_file(holeEtching.py ${CMAKE_CURRENT_BINARY_DIR}/holeEtching.py COPYONLY)
configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)

add_dependencies(ViennaPS_Examples ${PROJECT_NAME}.exe)
viennacore_setup_bat(${PROJECT_NAME}.exe ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
