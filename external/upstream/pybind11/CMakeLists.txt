# ##################################################################################################
# Check pybind11 Dependency
# ##################################################################################################
# If the path to an installation was provided by the user
if(DEFINED pybind11_DIR AND NOT DEFINED CACHE{pybind11_DIR})
  # Required for windows to convert backslashes to forward slashes
  file(TO_CMAKE_PATH "${pybind11_DIR}" pybind11_DIR)
  set(pybind11_DIR
      "${pybind11_DIR}"
      CACHE PATH "Path to pybind11 installation" FORCE)
  find_package(pybind11 REQUIRED PATHS ${pybind11_FOUND} NO_DEFAULT_PATH)
  message(STATUS "pybind11_DIR: ${pybind11_DIR}")
  add_library(pybind11_external INTERFACE) # dummy target
else()
  find_package(pybind11 QUIET)
  if(pybind11_FOUND)
    message(STATUS "Found pybind11 at ${pybind11_DIR}.")
    add_library(pybind11_external INTERFACE) # dummy target
  else()
    message(
      STATUS
        "Couldn't find pybind11. Dependency will be built with the buildDependencies target (e.g. `make buildDependencies`)."
    )

    # Get the number of cores
    include(ProcessorCount)
    ProcessorCount(NPROC)

    ExternalProject_Add(
      pybind11_external
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      # Hash of tags/v2.6.2
      GIT_TAG # v2.6.2
              8de7772cc72daca8e947b79b83fea46214931604
      # GIT_SHALLOW 1
      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
                 # -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                 -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                 -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
                 -DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}
                 -DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}
                 # -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
      BUILD_COMMAND ${CMAKE_COMMAND} --build . --parallel ${NPROC}
      # # No install needed INSTALL_COMMAND
      INSTALL_COMMAND ${CMAKE_COMMAND} --install .
      # "cmake --install build"
      UPDATE_COMMAND ""
      USES_TERMINAL_DOWNLOAD 1
      USES_TERMINAL_UPDATE 1
      GIT_PROGRESS 1)

    # Get install and build directory
    ExternalProject_Get_Property(pybind11_external INSTALL_DIR BINARY_DIR)
    set(pybind11_DIR "${INSTALL_DIR}")
    # add to buildDependencies target
    add_dependencies(buildDependencies pybind11_external)

  endif()
  set(pybind11_DIR
      "${pybind11_DIR}"
      CACHE PATH "Path to pybind11 installation" FORCE)
endif()
