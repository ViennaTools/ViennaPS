project(removeStrayPoints LANGUAGES CXX)

# TODO: This test gets stuck on MSVC for some reason...
if(NOT MSVC)
  add_executable(${PROJECT_NAME} "removeStrayPoints.cpp")
  target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)

  add_dependencies(ViennaPS_Tests ${PROJECT_NAME})
  add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
endif()
