add_custom_target(ViennaPS_Tests ALL)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/tests-bin>)

if(WIN32
   AND NOT VIENNAPS_SYSTEM_VIENNARAY
   AND NOT VIENNARAY_SYSTEM_EMBREE)
  setup_embree_env(ViennaPS_Tests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

if(WIN32
   AND NOT VIENNAPS_SYSTEM_VIENNARAY
   AND NOT VIENNARAY_SYSTEM_TBB)
  setup_tbb_env(ViennaPS_Tests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

if(WIN32
   AND NOT VIENNAPS_SYSTEM_VIENNALS
   AND NOT VIENNALS_SYSTEM_VTK)
  setup_vtk_env(ViennaPS_Tests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

file(
  GLOB entries
  LIST_DIRECTORIES true
  "*")

foreach(entry ${entries})
  if(NOT IS_DIRECTORY ${entry})
    continue()
  endif()

  message(STATUS "[ViennaPS] Adding Test ${entry}")
  add_subdirectory(${entry})
endforeach()
