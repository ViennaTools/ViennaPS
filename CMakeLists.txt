cmake_minimum_required(VERSION 3.4)

project(
  "ViennaPS"
  VERSION 0.0.1
  LANGUAGES CXX)


set(ViennaHRLE_DIR $ENV{VIENNAHRLE_DIR})
set(ViennaLS_DIR $ENV{VIENNALS_DIR})
#
# SET(ViennaHRLE_DIR "${ViennaHRLE_DIR}")
# SET(ViennaLS_DIR "${ViennaLS_DIR}")

find_package(ViennaHRLE REQUIRED)
find_package(ViennaLS REQUIRED)

set(VIENNAPS_INCLUDE_DIRS "${VIENNALS_INCLUDE_DIRS}" "${${PROJECT_NAME}_SOURCE_DIR}/include" "${${PROJECT_NAME}_SOURCE_DIR}/include/cellSet")


# VTK
if(NOT VTK_DIR)
    set(VTK_DIR $ENV{VTK_DIR})
endif(NOT VTK_DIR)

find_package(VTK REQUIRED)
if(VTK_FOUND)
  message(STATUS "Found VTK")
  add_definitions(-DVIENNALS_USE_VTK)
  include(${VTK_USE_FILE})
  # only link needed vtk libraries
  set(VTK_LIBRARIES vtksys;vtkIOCore;vtkexpat;vtklz4;vtkzlib;vtklzma;vtkdoubleconversion;vtkCommonMisc;vtkCommonSystem;vtkIOXML)
  list(APPEND VIENNAPS_LIBRARIES ${VTK_LIBRARIES})
else(VTK_FOUND)
  message(FATAL_ERROR "No VTK install found: Building without VTK.")
endif(VTK_FOUND)

#################################################
# BUILD TESTS (Smaller examples for testing)
#################################################
option(VIENNAPS_BUILD_TESTS "Build tests." OFF)
if(VIENNAPS_BUILD_TESTS)
  add_subdirectory(Tests)
endif(VIENNAPS_BUILD_TESTS)
