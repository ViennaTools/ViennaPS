project(stackEtching LANGUAGES CXX)

set(PROGRAM_NAME stackEtching)
add_executable(${PROGRAM_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROGRAM_NAME} PRIVATE ViennaPS)

configure_file(stackEtching.py ${CMAKE_CURRENT_BINARY_DIR}/stackEtching.py COPYONLY)
configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)

add_dependencies(ViennaPS_Examples ${PROGRAM_NAME})
viennacore_setup_bat(${PROGRAM_NAME} ${VIENNAPS_ARTIFACTS_DIRECTORY})
