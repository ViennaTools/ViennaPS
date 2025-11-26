project(testFluxModel LANGUAGES CXX)

add_gpu_executable(${PROJECT_NAME} target_name "${PROJECT_NAME}.cpp")
# target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)

add_dependencies(ViennaPS_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
