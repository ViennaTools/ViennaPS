cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# lowercase viennaPS is the python export
project("ViennaPSPython" VERSION ${VIENNAPS_VERSION})

include(GNUInstallDirs)

message(STATUS "ViennaPS version: ${VIENNAPS_VERSION}")

# include viennaPs
set(ViennaPS_DIR "${ViennaPS_BINARY_DIR}")

set(VIENNAPS_PYTHON_SOURCE "pyWrap.cpp")
set(VIENNAPS_PYTHON_MODULE_NAME "viennaps")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(PYBIND11_PYTHON_VERSION
    3
    CACHE STRING "Python version")

find_package(ViennaPS REQUIRED PATHS ${ViennaPS_DIR} NO_DEFAULT_PATH)
list(PREPEND VIENNAPS_INCLUDE_DIRS ${VIENNAPS_BUILD_INCLUDE_DIRS})
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})

# make 3.6 python module define dimension for build
set(VIENNAPS_PYTHON_MODULE_NAME_2D "${VIENNAPS_PYTHON_MODULE_NAME}2d")

pybind11_add_module(${VIENNAPS_PYTHON_MODULE_NAME_2D} ${VIENNAPS_PYTHON_SOURCE})
target_include_directories(${VIENNAPS_PYTHON_MODULE_NAME_2D} PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(${VIENNAPS_PYTHON_MODULE_NAME_2D} PRIVATE ${VIENNAPS_LIBRARIES})
# define compile dimension
target_compile_definitions(${VIENNAPS_PYTHON_MODULE_NAME_2D}
                            PRIVATE -DVIENNAPS_PYTHON_DIMENSION=2)

install(TARGETS ${VIENNAPS_PYTHON_MODULE_NAME_2D} LIBRARY DESTINATION ${ViennaPS_BINARY_DIR})

# define dimension for build
set(VIENNAPS_PYTHON_MODULE_NAME_3D "${VIENNAPS_PYTHON_MODULE_NAME}3d")

pybind11_add_module(${VIENNAPS_PYTHON_MODULE_NAME_3D} ${VIENNAPS_PYTHON_SOURCE})
target_include_directories(${VIENNAPS_PYTHON_MODULE_NAME_3D} PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(${VIENNAPS_PYTHON_MODULE_NAME_3D} PRIVATE ${VIENNAPS_LIBRARIES})
# define compile dimension
target_compile_definitions(${VIENNAPS_PYTHON_MODULE_NAME_3D}
                            PRIVATE -DVIENNAPS_PYTHON_DIMENSION=3)

install(TARGETS ${VIENNAPS_PYTHON_MODULE_NAME_3D} LIBRARY DESTINATION ${ViennaPS_BINARY_DIR})