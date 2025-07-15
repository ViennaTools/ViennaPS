project(customParticleDistribution LANGUAGES CXX)

add_executable("${PROJECT_NAME}" "${PROJECT_NAME}.cpp")
target_link_libraries("${PROJECT_NAME}" PRIVATE ViennaPS)

add_dependencies(ViennaPS_Examples ${PROJECT_NAME})
viennacore_setup_bat(${PROJECT_NAME} ${VIENNAPS_ARTIFACTS_DIRECTORY})
