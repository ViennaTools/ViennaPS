project(boschProcess LANGUAGES CXX)

add_executable(boschProcess "boschProcess.cpp")
target_link_libraries(boschProcess PRIVATE ViennaPS)

add_executable(boschEmulate "boschProcessEmulate.cpp")
target_link_libraries(boschEmulate PRIVATE ViennaPS)

add_dependencies(ViennaPS_Examples boschEmulate)
add_dependencies(ViennaPS_Examples boschProcess)

viennacore_setup_bat(boschProcess ${VIENNAPS_ARTIFACTS_DIRECTORY})
viennacore_setup_bat(boschEmulate ${VIENNAPS_ARTIFACTS_DIRECTORY})

configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)
