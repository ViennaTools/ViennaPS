project(postProcessingBenchmark LANGUAGES CXX)

CPMAddPackage(
  NAME nanoflann
  GIT_REPOSITORY "https://github.com/jlblancoc/nanoflann.git"
  GIT_TAG 1.12.1)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS nanoflann)

add_dependencies(ViennaPS_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
