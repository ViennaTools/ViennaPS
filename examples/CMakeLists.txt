add_custom_target(ViennaPS_Examples ALL)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/examples>)
if(WIN32)
  viennacore_setup_embree_env(ViennaPS_Examples ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  viennacore_setup_vtk_env(ViennaPS_Examples ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  viennacore_setup_tbb_env(ViennaPS_Examples ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

viennacore_add_subdirs(${CMAKE_CURRENT_LIST_DIR})
