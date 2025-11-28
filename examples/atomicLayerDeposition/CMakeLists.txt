project(atomicLayerDeposition LANGUAGES CXX)

viennaps_add_example(${PROJECT_NAME} "${PROJECT_NAME}.cpp")

configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)
