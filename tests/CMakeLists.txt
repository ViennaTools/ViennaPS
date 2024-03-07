add_custom_target(ViennaPS_Tests ALL)

include("${ViennaRay_SOURCE_DIR}/cmake/embree.cmake")

# TODO: Setup VTK Env as well
setup_embree_env(ViennaPS_Tests "tests")

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
