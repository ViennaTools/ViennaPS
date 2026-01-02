cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(
  ViennaPS
  LANGUAGES CXX C
  VERSION 4.1.3)

# --------------------------------------------------------------------------------------------------------
# Library switches
# --------------------------------------------------------------------------------------------------------

option(VIENNAPS_PRECOMPILE_HEADERS "Enable precompiled headers" OFF)
option(VIENNAPS_STATIC_BUILD "Build ViennaPS as static library" OFF)

option(VIENNAPS_USE_GPU "Enable GPU support" OFF)

option(VIENNAPS_BUILD_EXAMPLES "Build examples" OFF)
option(VIENNAPS_BUILD_TESTS "Build tests" OFF)

option(VIENNAPS_BUILD_PYTHON "Build python bindings" OFF)
option(VIENNAPS_PACKAGE_PYTHON "Build python bindings with intent to publish wheel" OFF)

# --------------------------------------------------------------------------------------------------------
# Library options
# --------------------------------------------------------------------------------------------------------

set(VIENNAPS_LOOKUP_DIRS
    ""
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

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES CXX_STANDARD 20
             CXX_EXTENSIONS OFF
             CXX_STANDARD_REQUIRED ON
             WINDOWS_EXPORT_ALL_SYMBOLS ON)

# --------------------------------------------------------------------------------------------------------
# Include directories
# --------------------------------------------------------------------------------------------------------

# Generate version header
configure_file("${PROJECT_SOURCE_DIR}/cmake/psVersion.hpp.in"
               "${PROJECT_SOURCE_DIR}/include/viennaps/psVersion.hpp" @ONLY)

target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/viennaps>
                            $<INSTALL_INTERFACE:include/viennaps-${PROJECT_VERSION}>)

# --------------------------------------------------------------------------------------------------------
# Setup Dependencies
# --------------------------------------------------------------------------------------------------------
include("cmake/cpm.cmake")

CPMAddPackage(
  NAME ViennaCore
  VERSION 1.8.0
  GIT_TAG msvc
  GIT_REPOSITORY "https://github.com/ViennaTools/ViennaCore"
  EXCLUDE_FROM_ALL ${VIENNAPS_BUILD_PYTHON}
  OPTIONS "VIENNACORE_USE_GPU ${VIENNAPS_USE_GPU}")

CPMAddPackage(
  NAME PackageProject
  VERSION 1.13.0
  GIT_REPOSITORY "https://github.com/TheLartians/PackageProject.cmake")

CPMFindPackage(
  NAME ViennaRay
  VERSION 3.9.1
  GIT_TAG refactor-cuda-setup
  GIT_REPOSITORY "https://github.com/ViennaTools/ViennaRay"
  EXCLUDE_FROM_ALL ${VIENNAPS_BUILD_PYTHON}
  OPTIONS "VIENNARAY_USE_GPU ${VIENNAPS_USE_GPU}")

CPMFindPackage(
  NAME ViennaLS
  VERSION 5.3.0
  GIT_TAG master
  GIT_REPOSITORY "https://github.com/ViennaTools/ViennaLS"
  EXCLUDE_FROM_ALL ${VIENNAPS_BUILD_PYTHON})

CPMFindPackage(
  NAME ViennaCS
  VERSION 1.1.1
  GIT_REPOSITORY "https://github.com/ViennaTools/ViennaCS"
  EXCLUDE_FROM_ALL ${VIENNAPS_BUILD_PYTHON})

target_link_libraries(${PROJECT_NAME} INTERFACE ViennaTools::ViennaCore ViennaTools::ViennaLS
                                                ViennaTools::ViennaRay ViennaTools::ViennaCS)

# --------------------------------------------------------------------------------------------------------
# Setup Shared/Static Library
# --------------------------------------------------------------------------------------------------------

if(VIENNAPS_PRECOMPILE_HEADERS)
  set(VIENNAPS_LINKAGE SHARED)

  if(VIENNAPS_STATIC_BUILD)
    set(VIENNAPS_LINKAGE STATIC)
  endif()

  set(LIB_NAME "viennaps")
  add_library(${LIB_NAME} ${VIENNAPS_LINKAGE})

  target_link_libraries(${LIB_NAME} PRIVATE ${PROJECT_NAME})
  target_compile_definitions(${LIB_NAME} PRIVATE VIENNAPS_USE_PRECOMPILED)
  target_sources(${LIB_NAME} PRIVATE "lib/specializationsMain.cpp" "lib/specializationsModels.cpp"
                                     "lib/specializationsGeometries.cpp")

  set_target_properties(
    ${LIB_NAME}
    PROPERTIES VERSION ${PROJECT_VERSION} # full x.y.z appears in filename
               SOVERSION ${PROJECT_VERSION_MAJOR} # major version for SONAME
               WINDOWS_EXPORT_ALL_SYMBOLS ON)

  target_link_libraries(${PROJECT_NAME} INTERFACE ${LIB_NAME})
  target_compile_definitions(${PROJECT_NAME} INTERFACE VIENNAPS_USE_PRECOMPILED)

  install(
    TARGETS ${LIB_NAME}
    EXPORT ViennaPSTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)
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
# Setup Examples
# --------------------------------------------------------------------------------------------------------

include("cmake/ViennaPSUtils.cmake")

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
  DEPENDENCIES "ViennaCore;ViennaLS;ViennaRay;ViennaCS;OpenMP")
