cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(
  ViennaPS
  LANGUAGES CXX
  VERSION 2.0.0)

# --------------------------------------------------------------------------------------------------------
# Library options
# --------------------------------------------------------------------------------------------------------

option(VIENNAPS_BUILD_APPLICATION "Build application" OFF)
option(VIENNAPS_BUILD_PYTHON "Build python bindings" OFF)

option(VIENNAPS_BUILD_EXAMPLES "Build examples" OFF)
option(VIENNAPS_BUILD_TESTS "Build tests" ON) # TODO: Default off

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
  # https://learn.microsoft.com/cpp/c-runtime-library/math-constants TODO: In case C++20 is adpoted
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
  NAME PackageProject
  VERSION 1.11.1
  GIT_REPOSITORY "https://github.com/TheLartians/PackageProject.cmake")

CPMFindPackage(
  NAME ViennaRay
  GIT_TAG 31add75 # TODO: Create new tag and use VERSION
  GIT_REPOSITORY "https://github.com/ViennaTools/ViennaRay")

CPMFindPackage(
  NAME ViennaLS
  VERSION 3.0.0
  GIT_REPOSITORY "https://github.com/ViennaTools/ViennaLS")

target_link_libraries(${PROJECT_NAME} INTERFACE ViennaTools::ViennaRay ViennaTools::ViennaLS)

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
# Setup Python Bindings
# --------------------------------------------------------------------------------------------------------

if(VIENNAPS_BUILD_PYTHON)
  message(STATUS "[ViennaPS] Building Python Bindings")
  add_subdirectory(python)
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
  DEPENDENCIES "ViennaLS;ViennaRay")
