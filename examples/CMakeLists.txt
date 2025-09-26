if(NOT VIENNACORE_USE_OPENMP)
  message(SEND_ERROR "ViennaPS examples are only available with OpenMP support.")
  return()
endif()

add_custom_target(ViennaPS_Examples ALL)

set(VIENNAPS_ARTIFACTS_DIRECTORY $<1:${PROJECT_BINARY_DIR}/artifacts>)

if(WIN32)
  viennacore_setup_embree_env(ViennaPS_Examples ${VIENNAPS_ARTIFACTS_DIRECTORY})
  viennacore_setup_vtk_env(ViennaPS_Examples ${VIENNAPS_ARTIFACTS_DIRECTORY})
  viennacore_setup_tbb_env(ViennaPS_Examples ${VIENNAPS_ARTIFACTS_DIRECTORY})
endif()

# viennacore_add_subdirs(${CMAKE_CURRENT_LIST_DIR})
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/holeEtching)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/trenchDeposition)