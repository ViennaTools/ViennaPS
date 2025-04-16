project(sputterDeposition LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)

add_executable(${PROJECT_NAME}3D "${PROJECT_NAME}3D.cpp")
target_link_libraries(${PROJECT_NAME}3D PRIVATE ViennaPS)

configure_file(config2D.txt ${CMAKE_CURRENT_BINARY_DIR}/config2D.txt COPYONLY)
configure_file(config3D.txt ${CMAKE_CURRENT_BINARY_DIR}/config3D.txt COPYONLY)
configure_file(rates2D.csv ${CMAKE_CURRENT_BINARY_DIR}/rates2D.csv COPYONLY)
configure_file(rates3D.csv ${CMAKE_CURRENT_BINARY_DIR}/rates3D.csv COPYONLY)
configure_file(generateRates.py ${CMAKE_CURRENT_BINARY_DIR}/generateRates.py COPYONLY)

add_dependencies(ViennaPS_Examples ${PROJECT_NAME})
viennacore_setup_bat(${PROJECT_NAME} ${VIENNAPS_ARTIFACTS_DIRECTORY})

add_dependencies(ViennaPS_Examples ${PROJECT_NAME}3D)
viennacore_setup_bat(${PROJECT_NAME}3D ${VIENNAPS_ARTIFACTS_DIRECTORY})
