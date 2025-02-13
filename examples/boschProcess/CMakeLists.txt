project(boschProcess LANGUAGES CXX)

add_executable(boschProcessSimulate "boschProcessSimulate.cpp")
target_link_libraries(boschProcessSimulate PRIVATE ViennaPS)

add_executable(boschProcessEmulate "boschProcessEmulate.cpp")
target_link_libraries(boschProcessEmulate PRIVATE ViennaPS)

add_dependencies(ViennaPS_Examples boschProcessEmulate)
add_dependencies(ViennaPS_Examples boschProcessSimulate)

viennacore_setup_bat(boschProcessSimulate ${VIENNAPS_ARTIFACTS_DIRECTORY})
viennacore_setup_bat(boschProcessEmulate ${VIENNAPS_ARTIFACTS_DIRECTORY})

configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)
