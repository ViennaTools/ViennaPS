project(fluxEngines LANGUAGES CXX)

if(VIENNAPS_USE_GPU)
  add_gpu_executable(${PROJECT_NAME} target_name "${PROJECT_NAME}.cpp")
else()
  add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
  target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)
endif()

add_dependencies(ViennaPS_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
