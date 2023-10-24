# external project base directory for all projects in this hierarchy
set_property(DIRECTORY PROPERTY EP_BASE ${DEPENDENCIES_DIR})

add_subdirectory(viennals)

add_subdirectory(viennaray)

if(VIENNAPS_BUILD_PYTHON)
  add_subdirectory(pybind11)
endif()
