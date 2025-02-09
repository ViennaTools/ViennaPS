project(boschProcess LANGUAGES CXX)

add_executable(boschProcess "boschProcess.cpp")
target_link_libraries(boschProcess PRIVATE ViennaPS)

add_executable(boschProcessEmulate "boschProcessEmulate.cpp")
target_link_libraries(boschProcessEmulate PRIVATE ViennaPS)

add_dependencies(ViennaPS_Examples boschProcessEmulate)
add_dependencies(ViennaPS_Examples boschProcess)

viennacore_setup_bat(boschProcess ${VIENNAPS_ARTIFACTS_DIRECTORY})
viennacore_setup_bat(boschProcessEmulate ${VIENNAPS_ARTIFACTS_DIRECTORY})

configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)
