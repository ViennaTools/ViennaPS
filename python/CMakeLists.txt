cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(ViennaPS_Python LANGUAGES CXX)

# --------------------------------------------------------------------------------------------------------
# Setup Target
# --------------------------------------------------------------------------------------------------------

add_custom_target(${PROJECT_NAME} ALL)

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
# Setup 2D-Bindings
# --------------------------------------------------------------------------------------------------------

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${VIENNAPS_LIBRARY_OUTPUT_DIR}/viennaps2d)
set(VIENNAPS_PYTHON_MODULE_NAME_2D "_${VIENNAPS_PYTHON_MODULE_NAME}2d")

pybind11_add_module("${VIENNAPS_PYTHON_MODULE_NAME_2D}" "pyWrap.cpp")
add_dependencies(${PROJECT_NAME} ${VIENNAPS_PYTHON_MODULE_NAME_2D})

target_link_libraries(${VIENNAPS_PYTHON_MODULE_NAME_2D} PUBLIC ViennaPS)
target_compile_definitions(
  ${VIENNAPS_PYTHON_MODULE_NAME_2D}
  PRIVATE -DVIENNAPS_PYTHON_DIMENSION=2 -DVIENNAPS_MODULE_NAME=${VIENNAPS_PYTHON_MODULE_NAME_2D})

set(MODULE_NAME ${VIENNAPS_PYTHON_MODULE_NAME_2D})
configure_file(__init__.py.in ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py)

# --------------------------------------------------------------------------------------------------------
# Setup 3D-Bindings
# --------------------------------------------------------------------------------------------------------

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${VIENNAPS_LIBRARY_OUTPUT_DIR}/viennaps3d)
set(VIENNAPS_PYTHON_MODULE_NAME_3D "_${VIENNAPS_PYTHON_MODULE_NAME}3d")

pybind11_add_module("${VIENNAPS_PYTHON_MODULE_NAME_3D}" "pyWrap.cpp")
add_dependencies(${PROJECT_NAME} ${VIENNAPS_PYTHON_MODULE_NAME_3D})

target_link_libraries(${VIENNAPS_PYTHON_MODULE_NAME_3D} PUBLIC ViennaPS)
target_compile_definitions(
  ${VIENNAPS_PYTHON_MODULE_NAME_3D}
  PRIVATE -DVIENNAPS_PYTHON_DIMENSION=3 -DVIENNAPS_MODULE_NAME=${VIENNAPS_PYTHON_MODULE_NAME_3D})

set(MODULE_NAME ${VIENNAPS_PYTHON_MODULE_NAME_3D})
configure_file(__init__.py.in ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py)
