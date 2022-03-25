cmake_minimum_required(VERSION 3.4)

project(
  "ViennaPS"
  VERSION 0.0.1
  LANGUAGES CXX)

include(GNUInstallDirs)

# set default build type
SET(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(ViennaHRLE_DIR $ENV{VIENNAHRLE_DIR})
set(ViennaLS_DIR $ENV{VIENNALS_DIR})
set(ViennaRay_DIR $ENV{VIENNARAY_DIR})

find_package(ViennaHRLE REQUIRED)
find_package(ViennaLS REQUIRED)
find_package (ViennaRay REQUIRED)

list(APPEND VIENNAPS_LIBRARIES ${VIENNARAY_LIBRARIES})

set(
    VIENNAPS_INCLUDE_DIRS 
    "${VIENNALS_INCLUDE_DIRS}" 
    "${VIENNARAY_INCLUDE_DIRS}"
    "${${PROJECT_NAME}_SOURCE_DIR}/include" 
    "${${PROJECT_NAME}_SOURCE_DIR}/include/cellSet" 
    )


# VTK
if(NOT VTK_DIR)
    set(VTK_DIR $ENV{VTK_DIR})
endif(NOT VTK_DIR)

find_package(VTK REQUIRED)
if(VTK_FOUND)
  message(STATUS "Found VTK")
  add_compile_definitions(VIENNAPS_USE_VTK)

  if (VTK_VERSION VERSION_LESS "8.90")
    message(STATUS "Using old VTK Version: ${VTK_VERSION}")
    include(${VTK_USE_FILE})
  endif()
  # only link needed vtk libraries
  # set(VTK_LIBRARIES vtksys;vtkIOCore;vtkexpat;vtklz4;vtkzlib;vtklzma;vtkdoubleconversion;vtkCommonMisc;vtkCommonSystem;vtkIOXML)
  list(APPEND VIENNAPS_LIBRARIES ${VTK_LIBRARIES})
else(VTK_FOUND)
  message(FATAL_ERROR "No VTK install found.")
endif(VTK_FOUND)

#################################################
# BUILD TESTS (Smaller examples for testing)
#################################################
option(VIENNAPS_BUILD_TESTS "Build tests." OFF)
if(VIENNAPS_BUILD_TESTS)
  add_subdirectory(Tests)
endif(VIENNAPS_BUILD_TESTS)

#################################################
# BUILD EXAMPLES 
#################################################
option(VIENNAPS_BUILD_EXAMPLES "Build examples." OFF)
if(VIENNAPS_BUILD_EXAMPLES)
  add_subdirectory(Examples)
endif(VIENNAPS_BUILD_EXAMPLES)

#################################################
# INSTALL
#################################################
# install config files
# locations are provided by GNUInstallDirs
add_library(${PROJECT_NAME} INTERFACE)

# set the correct paths for installation
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}/")
set(LOCAL_CONFIG_DIR "lib/cmake/${PROJECT_NAME}")

# Adding the install interface generator expression makes sure that the include
# files are installed to the proper location (provided by GNUInstallDirs)
set(VIENNAPS_BUILD_INCLUDE_DIRS "${${PROJECT_NAME}_SOURCE_DIR}/include")
target_include_directories(
  ${PROJECT_NAME}
  INTERFACE $<BUILD_INTERFACE:${VIENNAPS_BUILD_INCLUDE_DIRS}>
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION}
                               COMPATIBILITY AnyNewerVersion)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION
  ${LOCAL_CONFIG_DIR})

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${LOCAL_CONFIG_DIR})


install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${LOCAL_CONFIG_DIR})

# install include files
file(GLOB_RECURSE HEADER_FILES "${PROJECT_SOURCE_DIR}/include/*.hpp")
install(FILES ${HEADER_FILES} DESTINATION include)