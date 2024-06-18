project(atomicLayerDeposition LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)

configure_file(atomicLayerDeposition.py ${CMAKE_CURRENT_BINARY_DIR}/atomicLayerDeposition.py
               COPYONLY)
configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)

add_dependencies(ViennaPS_Examples ${PROJECT_NAME})
viennacore_setup_bat(${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR})
