project(surfaceChemistry LANGUAGES CXX)

viennaps_add_example(${PROJECT_NAME} "${PROJECT_NAME}.cpp")

# the reaction files are the input to the example, so they travel with it
file(COPY ${CMAKE_CURRENT_LIST_DIR}/reactions
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
