# Check if the variable is a cache variable
get_property(
  is_cached
  CACHE ViennaLS_DIR
  PROPERTY TYPE)
# If it is not from cache and defined at this point, the variable was passed as a -D command line
# argument
if("${is_cached}" STREQUAL "" AND DEFINED ViennaLS_DIR)
  # Required for windows to convert backslashes to forward slashes
  file(TO_CMAKE_PATH "${ViennaLS_DIR}" ViennaLS_DIR)
  set(ViennaLS_DIR
      "${ViennaLS_DIR}"
      CACHE PATH "Path to ViennaLS installation" FORCE)
  # Now look for the package at the provided location
  find_package(ViennaLS REQUIRED PATHS ${ViennaLS_DIR} NO_DEFAULT_PATH)
  if(ViennaLS_FOUND AND NOT ${VIENNALS_USE_VTK})
    message(FATAL_ERROR "ViennaLS was built without VTK support!")
  endif()
  message(STATUS "Found ViennaLS at ${ViennaLS_DIR}")
  add_library(viennals_external INTERFACE) # dummy target
else()
  # First try to locate any local build
  find_package(ViennaLS QUIET PATHS "${DEPENDENCIES_DIR}/Install/viennals_external" NO_DEFAULT_PATH)

  # If that fails look for a system installation
  if(NOT ViennaLS_FOUND)
    find_package(ViennaLS QUIET)
  endif()

  if(ViennaLS_FOUND AND NOT ${VIENNALS_USE_VTK})
    set(ViennaLS_FOUND FALSE)
    message(
      STATUS "An installation of ViennaLS was found, but it was built without support for VTK.")
  endif()
  if(ViennaLS_FOUND)
    message(STATUS "Found ViennaLS at ${ViennaLS_DIR}.")
    add_library(viennals_external INTERFACE) # dummy target
  else()
    message(
      STATUS
        "Couldn't find ViennaLS. Dependency will be built with the buildDependencies target (e.g. `make buildDependencies`)."
    )
    ExternalProject_Add(
      viennals_external
      GIT_REPOSITORY https://github.com/ViennaTools/ViennaLS.git
      GIT_SHALLOW 1
      INSTALL_DIR "${DEPENDENCIES_DIR}/Install/viennals_external"
      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
                 -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                 -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                 -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
                 -DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}
                 -DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}
                 -DVIENNALS_BUILD_EXAMPLES=OFF
                 -DVIENNALS_BUILD_TESTS=OFF
                 -DVIENNALS_USE_VTK=ON
      BUILD_COMMAND ${CMAKE_COMMAND} --build . --target buildDependencies all
      USES_TERMINAL_DOWNLOAD 1
      USES_TERMINAL_UPDATE 1
      GIT_PROGRESS 1
      EXCLUDE_FROM_ALL TRUE)

    # Get install and build directory
    ExternalProject_Get_Property(viennals_external INSTALL_DIR)

    set(ViennaLS_DIR "${INSTALL_DIR}")

    # add to buildDependencies target
    add_dependencies(buildDependencies viennals_external)
  endif()
  set(ViennaLS_DIR
      ${ViennaLS_DIR}
      CACHE PATH "Path to ViennaLS installation" FORCE)
endif()
