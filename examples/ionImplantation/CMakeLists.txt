project(ionImplantation LANGUAGES CXX)

# Explicit Pearson IV moments from config
set(VIENNAPS_MODELDB_DIR "${CMAKE_SOURCE_DIR}/modeldb")

# Unified example (Explicit & Table-driven based on config file)
add_executable(${PROJECT_NAME} "ionImplantation.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)
target_compile_definitions(${PROJECT_NAME} PRIVATE "VIENNAPS_MODELDB_DIR=\"${VIENNAPS_MODELDB_DIR}\"")
add_dependencies(ViennaPS_Examples ${PROJECT_NAME})
viennacore_setup_bat(${PROJECT_NAME} ${VIENNAPS_ARTIFACTS_DIRECTORY})

# Copy config files into the build directory
configure_file(config.txt       ${CMAKE_CURRENT_BINARY_DIR}/config.txt       COPYONLY)
configure_file(config_default.txt ${CMAKE_CURRENT_BINARY_DIR}/config_default.txt COPYONLY)
