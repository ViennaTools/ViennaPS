cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(
  ViennaPS
  LANGUAGES CXX C
  VERSION 3.5.1)

# --------------------------------------------------------------------------------------------------------
# Library switches
# --------------------------------------------------------------------------------------------------------

option(VIENNAPS_USE_GPU "Enable GPU support" ON)
option(VIENNAPS_FORCE_GPU "Require GPU support" OFF)

option(VIENNAPS_BUILD_EXAMPLES "Build examples" OFF)
option(VIENNAPS_BUILD_TESTS "Build tests" OFF)

option(VIENNAPS_BUILD_PYTHON "Build python bindings" OFF)
option(VIENNAPS_PACKAGE_PYTHON "Build python bindings with intent to publish wheel" OFF)

# --------------------------------------------------------------------------------------------------------
# Library options
# --------------------------------------------------------------------------------------------------------

set(VIENNAPS_LOOKUP_DIRS
    ""
    # "${CMAKE_CURRENT_SOURCE_DIR}/../install/"
    CACHE STRING "Directories to account for when searching installed dependencies")

list(APPEND CMAKE_PREFIX_PATH ${VIENNAPS_LOOKUP_DIRS})

# --------------------------------------------------------------------------------------------------------
# Global CMake Configuration
# └ We depend on the vtk python package for our python build and re-use
#   their existing vtk libraries, which are not versioned. Because we (possibly) pull in VTK ourselves
#   during build time though, we disable the soname versioning so that we can easily set the rpath to
#   link against the vtk-python libs
# --------------------------------------------------------------------------------------------------------

if(VIENNAPS_PACKAGE_PYTHON)
  message(
    STATUS
      "[ViennaPS] Using remote packages due to 'VIENNAPS_PACKAGE_PYTHON', the build will take a long time!"
  )

  # CIBUILDWHEEL is absolutely cursed and won't work when this is turned on.
  # I don't know why...
  if(NOT VIENNAPS_IS_CI)
    set(CPM_DOWNLOAD_ALL ON)
  endif()

  set(VIENNAPS_BUILD_PYTHON ON)
endif()

if(VIENNAPS_BUILD_PYTHON)
  set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp:llvm /bigobj")
endif()

# --------------------------------------------------------------------------------------------------------
# Setup Library
# --------------------------------------------------------------------------------------------------------

add_library(${PROJECT_NAME} INTERFACE)
add_library(ViennaTools::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES CXX_STANDARD 17
             CXX_EXTENSIONS OFF
             CXX_STANDARD_REQUIRED ON
             WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(MSVC)
  # https://learn.microsoft.com/cpp/c-runtime-library/math-constants TODO: In case C++20 is adopted
  # any time soon: https://cppreference.com/w/cpp/numeric/constants

  target_compile_definitions(${PROJECT_NAME} INTERFACE _USE_MATH_DEFINES=1)
endif()

# --------------------------------------------------------------------------------------------------------
# Include directories
# --------------------------------------------------------------------------------------------------------

target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/viennaps>
                            $<INSTALL_INTERFACE:include/viennaps-${PROJECT_VERSION}>)

# --------------------------------------------------------------------------------------------------------
# Setup Dependencies
# --------------------------------------------------------------------------------------------------------
include("cmake/cpm.cmake")

CPMAddPackage(
  NAME ViennaCore
  VERSION 1.4.3
  GIT_REPOSITORY "https://github.com/ViennaTools/ViennaCore"
  EXCLUDE_FROM_ALL ${VIENNAPS_BUILD_PYTHON}
  OPTIONS "VIENNACORE_USE_GPU ${VIENNAPS_USE_GPU}" "VIENNACORE_FORCE_GPU ${VIENNAPS_FORCE_GPU}")

CPMAddPackage(
  NAME PackageProject
  VERSION 1.13.0
  GIT_REPOSITORY "https://github.com/TheLartians/PackageProject.cmake")

CPMFindPackage(
  NAME ViennaRay
  VERSION 3.4.1
  GIT_REPOSITORY "https://github.com/ViennaTools/ViennaRay"
  EXCLUDE_FROM_ALL ${VIENNAPS_BUILD_PYTHON}
  OPTIONS "VIENNARAY_USE_GPU ${VIENNAPS_USE_GPU}")

CPMFindPackage(
  NAME ViennaLS
  VERSION 4.3.2
  GIT_REPOSITORY "https://github.com/ViennaTools/ViennaLS"
  EXCLUDE_FROM_ALL ${VIENNAPS_BUILD_PYTHON})

CPMFindPackage(
  NAME ViennaCS
  VERSION 1.1.0
  GIT_REPOSITORY "https://github.com/ViennaTools/ViennaCS"
  EXCLUDE_FROM_ALL ${VIENNAPS_BUILD_PYTHON})

target_link_libraries(${PROJECT_NAME} INTERFACE ViennaTools::ViennaCore ViennaTools::ViennaLS
                                                ViennaTools::ViennaRay ViennaTools::ViennaCS)

# --------------------------------------------------------------------------------------------------------
# Setup Examples
# --------------------------------------------------------------------------------------------------------

if(VIENNAPS_BUILD_EXAMPLES)
  message(STATUS "[ViennaPS] Building Examples")
  add_subdirectory(examples)
endif()

# --------------------------------------------------------------------------------------------------------
# Setup Tests
# --------------------------------------------------------------------------------------------------------

if(VIENNAPS_BUILD_TESTS)
  message(STATUS "[ViennaPS] Building Tests")

  enable_testing()
  add_subdirectory(tests)
endif()

# --------------------------------------------------------------------------------------------------------
# Setup GPU
# --------------------------------------------------------------------------------------------------------

# If CUDA or OptiX is not found in ViennaCore, VIENNACORE_PTX_DIR is not set
if(VIENNAPS_USE_GPU AND VIENNACORE_PTX_DIR)
  message(STATUS "[ViennaPS] Enabled GPU Support")
  add_subdirectory(gpu)
else()
  message(STATUS "[ViennaPS] Disabled GPU Support")
  set(VIENNAPS_USE_GPU OFF)
endif()

# --------------------------------------------------------------------------------------------------------
# Setup Python Bindings
# --------------------------------------------------------------------------------------------------------

if(VIENNAPS_BUILD_PYTHON)
  message(STATUS "[ViennaPS] Building Python Bindings")
  add_subdirectory(python)
  return()
endif()

# --------------------------------------------------------------------------------------------------------
# Install Target
# --------------------------------------------------------------------------------------------------------

packageProject(
  NAME ${PROJECT_NAME} NAMESPACE ViennaTools
  VERSION ${PROJECT_VERSION}
  BINARY_DIR ${PROJECT_BINARY_DIR}
  INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/viennaps
  INCLUDE_DESTINATION include/viennaps-${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
  DEPENDENCIES "ViennaCore;ViennaLS;ViennaRay;ViennaCS")
