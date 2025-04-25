project(SiGePlasmaEtching LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)

configure_file(config_1pillar.txt ${CMAKE_CURRENT_BINARY_DIR}/config_1pillar.txt COPYONLY)
configure_file(config_5pillar.txt ${CMAKE_CURRENT_BINARY_DIR}/config_5pillar.txt COPYONLY)
configure_file(config_realGeo.txt ${CMAKE_CURRENT_BINARY_DIR}/config_realGeo.txt COPYONLY)
configure_file(10s_20mTorr_100nm_init.csv ${CMAKE_CURRENT_BINARY_DIR}/10s_20mTorr_100nm_init.csv COPYONLY)
configure_file(10s_20mTorr_100nm_final.csv ${CMAKE_CURRENT_BINARY_DIR}/10s_20mTorr_100nm_final.csv COPYONLY)
configure_file(exractSurface.ipynb ${CMAKE_CURRENT_BINARY_DIR}/exractSurface.ipynb COPYONLY)

add_dependencies(ViennaPS_Examples ${PROJECT_NAME})
viennacore_setup_bat(${PROJECT_NAME} ${VIENNAPS_ARTIFACTS_DIRECTORY})

