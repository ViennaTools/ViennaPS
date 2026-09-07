project(surfaceChemistry LANGUAGES CXX)

viennaps_add_example(${PROJECT_NAME} "${PROJECT_NAME}.cpp")

# a cyclic process: two reaction files and a four-phase list
viennaps_add_example(cyclicProcess "cyclicProcess.cpp")

# the reaction files are the input to the example, so they travel with it
file(COPY ${CMAKE_CURRENT_LIST_DIR}/reactions
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
