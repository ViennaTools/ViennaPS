project(plasmaEtchingEngines LANGUAGES CXX)

if(VIENNAPS_USE_GPU)
  viennaps_add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")

  add_dependencies(ViennaPS_Tests ${PROJECT_NAME})
  add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
endif()
