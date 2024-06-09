add_custom_target(ViennaPS_Examples ALL)

if(WIN32)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/examples>)
  setup_embree_env(ViennaPS_Examples ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  setup_vtk_env(ViennaPS_Examples ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  setup_tbb_env(ViennaPS_Examples ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

viennacore_add_subdirs(${CMAKE_CURRENT_LIST_DIR})
