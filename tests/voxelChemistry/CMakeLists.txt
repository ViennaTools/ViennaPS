project(voxelChemistry LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)

# The mechanisms are the ones the surfaceChemistry example ships, so the test
# runs the published format rather than a mechanism built in code -- which is
# the whole claim: both arms read the same file. Compiled in rather than passed
# as arguments, so the test is invoked bare like every other one here.
target_compile_definitions(
  ${PROJECT_NAME}
  PRIVATE
    VIENNAPS_MECHANISM_DIR="${CMAKE_CURRENT_LIST_DIR}/../../examples/surfaceChemistry/reactions")

add_dependencies(ViennaPS_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
