add_custom_target(ViennaPS_Examples ALL)

include("../cmake/bat.cmake")

set(VIENNAPS_EXAMPLES_LIB $<1:${PROJECT_BINARY_DIR}/example-libs>)

if(WIN32 AND NOT VIENNARAY_SYSTEM_EMBREE)
  setup_embree_env(ViennaPS_Examples ${VIENNAPS_EXAMPLES_LIB})
endif()

if(WIN32 AND NOT VIENNARAY_SYSTEM_TBB)
  setup_tbb_env(ViennaPS_Examples ${VIENNAPS_EXAMPLES_LIB})
endif()

if(WIN32 AND NOT VIENNALS_SYSTEM_VTK)
  setup_vtk_env(ViennaPS_Examples ${VIENNAPS_EXAMPLES_LIB})
endif()

file(
  GLOB entries
  LIST_DIRECTORIES true
  "*")

foreach(entry ${entries})
  if(NOT IS_DIRECTORY ${entry})
    continue()
  endif()

  message(STATUS "[ViennaPS] Adding Example ${entry}")
  add_subdirectory(${entry})
endforeach()
