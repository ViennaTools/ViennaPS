add_custom_target(ViennaPS_Examples ALL)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/examples-bin>)

if(WIN32
   AND NOT VIENNARAY_DISABLE_COPY
   AND NOT VIENNARAY_SYSTEM_EMBREE)
  setup_embree_env(ViennaPS_Examples ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

if(WIN32 AND NOT VIENNALS_SYSTEM_VTK)
  setup_vtk_env(ViennaPS_Examples ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
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
