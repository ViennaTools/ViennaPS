project(boschProcess LANGUAGES CXX)

add_executable(bosch "boschProcess.cpp")
target_link_libraries(bosch PRIVATE ViennaPS)

add_dependencies(ViennaPS_Examples bosch)
# viennacore_setup_bat(bosch ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)
