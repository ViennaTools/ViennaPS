cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project("ViennaPS" VERSION 1.0.0)

include(GNUInstallDirs)

# c++17 for inline constexpr variables
set(CMAKE_CXX_STANDARD "17")

# Set default build type
set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "${DEFAULT_BUILD_TYPE}"
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

# tell VS to export all symbols to its dll files
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS
      TRUE
      CACHE BOOL "Export all symbols")
  add_compile_definitions(_USE_MATH_DEFINES)
endif()

set(ViennaPS_BINARY_DIR
    ${PROJECT_BINARY_DIR}
    CACHE PATH "Path to local ViennaPS installation" FORCE)

# build dependencies
set(DEPENDENCIES_DIR ${CMAKE_SOURCE_DIR}/dependencies)

# Include all external dependencies
include(ExternalProject)
add_custom_target(buildDependencies)
add_subdirectory(external/upstream)

option(VIENNAPS_VERBOSE "Get verbose output and intermediate results." ON)
if(VIENNAPS_VERBOSE)
  add_compile_definitions(VIENNAPS_VERBOSE)
endif(VIENNAPS_VERBOSE)

# install config files locations are provided by GNUInstallDirs
add_library(${PROJECT_NAME} INTERFACE)

# set the correct paths for installation
set(LOCAL_CONFIG_DIR "lib/cmake/${PROJECT_NAME}")

# Adding the install interface generator expression makes sure that the include files are installed
# to the proper location (provided by GNUInstallDirs)
set(VIENNAPS_BUILD_INCLUDE_DIRS "${${PROJECT_NAME}_SOURCE_DIR}/include")
target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${VIENNAPS_BUILD_INCLUDE_DIRS}>
                            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" INSTALL_DESTINATION ${LOCAL_CONFIG_DIR})

# ##################################################################################################
# BUILD EXAMPLES
# ##################################################################################################
option(VIENNAPS_BUILD_EXAMPLES "Build examples." OFF)
if(VIENNAPS_BUILD_EXAMPLES)
  add_subdirectory(Examples)
endif(VIENNAPS_BUILD_EXAMPLES)

# ##################################################################################################
# INSTALL
# ##################################################################################################

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}_Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(
  EXPORT ${PROJECT_NAME}_Targets
  FILE ${PROJECT_NAME}Targets.cmake
  DESTINATION ${LOCAL_CONFIG_DIR})

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${LOCAL_CONFIG_DIR})

# install include files
file(GLOB_RECURSE HEADER_FILES "${PROJECT_SOURCE_DIR}/include/*.hpp")

install(FILES ${HEADER_FILES} DESTINATION "include")
