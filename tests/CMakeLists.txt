cmake_minimum_required(VERSION 3.14)

project(ViennaPSTests)

include("${CMAKE_SOURCE_DIR}/cmake/prepare.cmake")

# Point find_package to the binary directory instead of the install location
find_package(ViennaPS REQUIRED CONFIG PATHS ${ViennaPS_BINARY_DIR} NO_DEFAULT_PATH)

list(PREPEND VIENNAPS_INCLUDE_DIRS ${VIENNAPS_BUILD_INCLUDE_DIRS})

# Add subfolders to the include directories, since during installation the directory structure is
# flattened, but in the source tree - which we are using here - it is not.
list(PREPEND VIENNAPS_INCLUDE_DIRS ${VIENNAPS_BUILD_INCLUDE_DIRS}/cellSet
     ${VIENNAPS_BUILD_INCLUDE_DIRS}/geometries ${VIENNAPS_BUILD_INCLUDE_DIRS}/models
     ${VIENNAPS_BUILD_INCLUDE_DIRS}/compact)

set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL TRUE)

add_custom_target(buildTests DEPENDS ${VIENNAPS_DEPENDENCIES})
if(MSVC)
  target_compile_options(buildTests PRIVATE /bigobj)
endif()

subdirlist(TEST_DIRS ${CMAKE_CURRENT_LIST_DIR})

# add all tests
foreach(subdir ${TEST_DIRS})
  add_subdirectory(${subdir})
endforeach()
