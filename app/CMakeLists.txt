project(ViennaPSApplication LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(ViennaPS2D "application.cpp")
target_link_libraries(ViennaPS2D PRIVATE ViennaPS)
target_compile_definitions(ViennaPS2D PRIVATE VIENNAPS_APP_DIM=2)

add_executable(ViennaPS3D "application.cpp")
target_link_libraries(ViennaPS3D PRIVATE ViennaPS)
target_compile_definitions(ViennaPS3D PRIVATE VIENNAPS_APP_DIM=3)

configure_file(sampleConfig.txt ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sampleConfig.txt COPYONLY)
