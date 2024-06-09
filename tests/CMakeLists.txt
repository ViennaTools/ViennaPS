add_custom_target(ViennaPS_Tests ALL)

if(WIN32)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/tests>)
  setup_embree_env(ViennaPS_Tests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  setup_vtk_env(ViennaPS_Tests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  setup_tbb_env(ViennaPS_Tests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

viennacore_add_subdirs(${CMAKE_CURRENT_LIST_DIR})
