cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(ViennaPS_Python LANGUAGES CXX)

add_custom_target(${PROJECT_NAME} ALL)

# --------------------------------------------------------------------------------------------------------
# Global CMake Configuration
# └ As mentioned earlier we re-use the vtk python package libs, to do so we
#   set the rpath for our python modules to point to the vtkmodules folder.
# --------------------------------------------------------------------------------------------------------

set(CMAKE_MACOSX_RPATH ON)

set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

if(NOT APPLE)
  list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN")
  list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../vtkmodules")
  list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../viennals.libs")
  list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../viennaps.libs")
else()
  list(APPEND CMAKE_INSTALL_RPATH "@loader_path")
  list(APPEND CMAKE_INSTALL_RPATH "@loader_path/../vtkmodules")
  list(APPEND CMAKE_INSTALL_RPATH "@loader_path/../viennals.libs")
  list(APPEND CMAKE_INSTALL_RPATH "@loader_path/../viennaps.libs")
endif()

# --------------------------------------------------------------------------------------------------------
# Setup Dependencies
# --------------------------------------------------------------------------------------------------------

include("../cmake/cpm.cmake")

set(PYBIND11_FINDPYTHON ON)

CPMFindPackage(
  NAME pybind11
  VERSION 2.11.1
  GIT_REPOSITORY "https://github.com/pybind/pybind11")

# --------------------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------------------

set(VIENNAPS_PYTHON_MODULE_NAME "viennaps")
set(VIENNAPS_LIBRARY_OUTPUT_DIR ${CMAKE_BINARY_DIR})

if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(VIENNAPS_LIBRARY_OUTPUT_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  message(STATUS "[ViennaPS] Using Library Output Directory for bindings")
endif()

# --------------------------------------------------------------------------------------------------------
# Binding macro
# --------------------------------------------------------------------------------------------------------

macro(setup_binding NAME FLAGS)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${VIENNAPS_LIBRARY_OUTPUT_DIR}/${NAME})
  set(MODULE_NAME "_${NAME}")

  pybind11_add_module("${MODULE_NAME}" "pyWrap.cpp")
  add_dependencies(${PROJECT_NAME} ${MODULE_NAME})

  target_link_libraries(${MODULE_NAME} PUBLIC ViennaPS)
  target_compile_definitions(${MODULE_NAME} PRIVATE ${FLAGS} -DVIENNAPS_MODULE_NAME=${MODULE_NAME})

  configure_file(__init__.py.in ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py)

  install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION ${NAME})
  install(DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/" DESTINATION ${NAME})
endmacro()

# --------------------------------------------------------------------------------------------------------
# Setup Bindings
# --------------------------------------------------------------------------------------------------------

setup_binding(viennaps2d -DVIENNAPS_PYTHON_DIMENSION=2)
setup_binding(viennaps3d -DVIENNAPS_PYTHON_DIMENSION=3)

set(VIENNAPS_LIB_FOLDER "${VIENNAPS_LIBRARY_OUTPUT_DIR}/viennaps.libs")
set(VIENNAPS_USE_LIB_FOLDER OFF)

if(VIENNAPS_PACKAGE_PYTHON OR (NOT VIENNAPS_SYSTEM_VIENNARAY AND NOT VIENNARAY_SYSTEM_EMBREE))
  set(VIENNAPS_USE_LIB_FOLDER ON)
  setup_embree_env(${MODULE_NAME} "${VIENNAPS_LIB_FOLDER}")
endif()

if(VIENNAPS_PACKAGE_PYTHON OR (NOT VIENNAPS_SYSTEM_VIENNARAY AND NOT VIENNARAY_SYSTEM_TBB))
  set(VIENNAPS_USE_LIB_FOLDER ON)
  setup_tbb_env(${MODULE_NAME} "${VIENNAPS_LIB_FOLDER}")
endif()

if(VIENNAPS_USE_LIB_FOLDER)
  install(DIRECTORY "${VIENNAPS_LIB_FOLDER}" DESTINATION .)
endif()
