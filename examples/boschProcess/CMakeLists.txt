project(boschProcess LANGUAGES CXX)

add_executable(boschSimulate "boschProcess.cpp")
target_link_libraries(boschSimulate PRIVATE ViennaPS)
add_dependencies(ViennaPS_Examples boschSimulate)

add_executable(boschEmulate "boschProcessEmulate.cpp")
target_link_libraries(boschEmulate PRIVATE ViennaPS)
add_dependencies(ViennaPS_Examples boschEmulate)

configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)
