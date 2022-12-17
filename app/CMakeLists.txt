cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(ViennaPSApplication)

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
     ${VIENNAPS_BUILD_INCLUDE_DIRS}/Geometries ${VIENNAPS_BUILD_INCLUDE_DIRS}/Models
     ${VIENNAPS_BUILD_INCLUDE_DIRS}/Compact)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

option(VIENNAPS_USE_DOUBLE "Use double precision floating point numbers in application." OFF)
if(VIENNAPS_USE_DOUBLE)
  add_compile_definitions(VIENNAPS_USE_DOUBLE)
endif(VIENNAPS_USE_DOUBLE)

set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL TRUE)

add_custom_target(buildApplication)

add_executable(ViennaPS2D ${PROJECT_SOURCE_DIR}/Application2D.cpp)
target_include_directories(ViennaPS2D PUBLIC ${VIENNAPS_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
target_link_libraries(ViennaPS2D ${VIENNAPS_LIBRARIES})
add_dependencies(buildApplication ViennaPS2D)

add_executable(ViennaPS3D ${PROJECT_SOURCE_DIR}/Application3D.cpp)
target_include_directories(ViennaPS3D PUBLIC ${VIENNAPS_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
target_link_libraries(ViennaPS3D ${VIENNAPS_LIBRARIES})
add_dependencies(buildApplication ViennaPS3D)

configure_file(SampleConfig.txt ${CMAKE_BINARY_DIR}/SampleConfig.txt COPYONLY)
