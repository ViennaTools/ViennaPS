cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(ViennaPSExamples)

include("${CMAKE_SOURCE_DIR}/cmake/prepare.cmake")

find_package(ViennaPS CONFIG PATHS ${ViennaPS_BINARY_DIR})
if(NOT ViennaPS_FOUND)
  message(
    FATAL_ERROR
      "One of the dependencies hasn't been built yet and thus can't be used by the buildExamples target.\n\
  Try the following steps to fix the problem:\n\
  1. Turn off the buildExamples target by disabling the VIENNAPS_BUILD_EXAMPLES option when configuring the project\n\
  (e.g. with the `-D VIENNAPS_BUILD_EXAMPLES=OFF` flag)\n\
  2. Build the dependencies target: e.g. `make buildDependencies` when using Makefiles\n\
  3. Turn on the buildExamples target again by re-enabling the VIENNAPS_BUILD_EXAMPLES option\n\
  (e.g. with the `-D VIENNAPS_BUILD_EXAMPLES=ON` flag)\n\
  4. Now the configure step should pick up the built dependencies and proceed as intended.")
endif()

list(PREPEND VIENNAPS_INCLUDE_DIRS ${VIENNAPS_BUILD_INCLUDE_DIRS})

# Add subfolders to the include directories, since during installation the directory structure is
# flattened, but in the source tree - which we are using here - it is not.
list(PREPEND VIENNAPS_INCLUDE_DIRS ${VIENNAPS_BUILD_INCLUDE_DIRS}/CellSet
     ${VIENNAPS_BUILD_INCLUDE_DIRS}/Geometries ${VIENNAPS_BUILD_INCLUDE_DIRS}/Models)

# Exclude the buildExamples target from the `all` target
set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL TRUE)

add_custom_target(buildExamples)

subdirlist(EXAMPLE_DIRS ${CMAKE_CURRENT_LIST_DIR})

# add all examples
foreach(subdir ${EXAMPLE_DIRS})
  add_subdirectory(${subdir})
endforeach()
