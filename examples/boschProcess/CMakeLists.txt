project(boschProcess LANGUAGES CXX)

viennaps_add_example(boschProcessSimulate "boschProcessSimulate.cpp")

add_executable(boschProcessEmulate "boschProcessEmulate.cpp")
target_link_libraries(boschProcessEmulate PRIVATE ViennaPS)

add_dependencies(ViennaPS_Examples boschProcessEmulate)

viennacore_setup_bat(boschProcessEmulate ${VIENNAPS_ARTIFACTS_DIRECTORY})

configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)
