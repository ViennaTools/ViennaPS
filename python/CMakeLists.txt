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
  list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../viennaps.libs")
else()
  list(APPEND CMAKE_INSTALL_RPATH "@loader_path")
  list(APPEND CMAKE_INSTALL_RPATH "@loader_path/../vtkmodules")
  list(APPEND CMAKE_INSTALL_RPATH "@loader_path/../viennaps.libs")
endif()

# --------------------------------------------------------------------------------------------------------
# Setup Dependencies
# --------------------------------------------------------------------------------------------------------

include("../cmake/cpm.cmake")

set(PYBIND11_FINDPYTHON ON)

CPMFindPackage(
  NAME pybind11
  VERSION 3.0.0
  GIT_REPOSITORY "https://github.com/pybind/pybind11")

# --------------------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------------------

set(VIENNAPS_PYTHON_PACKAGE_NAME viennaps)
set(VIENNAPS_PYTHON_MODULE_NAME _core)
set(VIENNAPS_LIBRARY_OUTPUT_DIR ${CMAKE_BINARY_DIR})

if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(VIENNAPS_LIBRARY_OUTPUT_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  message(STATUS "[ViennaPS] Using Library Output Directory for bindings")
endif()

# --------------------------------------------------------------------------------------------------------
# Setup Bindings
# --------------------------------------------------------------------------------------------------------

pybind11_add_module(${VIENNAPS_PYTHON_MODULE_NAME} pyWrap.cpp)
target_link_libraries(${VIENNAPS_PYTHON_MODULE_NAME} PRIVATE ViennaPS)
target_compile_definitions(${VIENNAPS_PYTHON_MODULE_NAME}
                           PRIVATE VIENNAPS_MODULE_NAME=${VIENNAPS_PYTHON_MODULE_NAME})

# To prevent symbol clashes when multiple python scripts which use VTK modules are loaded, the modules have to be initialized here
if(VIENNALS_VTK_RENDERING)
  # We need to find VTK again here to get the proper module targets
  # TODO: Refactor to avoid multiple finds
  CPMFindPackage(
    NAME VTK
    GIT_TAG v9.3.1
    VERSION 9.0.0
    GIT_REPOSITORY "https://gitlab.kitware.com/vtk/vtk")

  # These modules need to be initialized for VTK rendering to work properly
  vtk_module_autoinit(TARGETS ${VIENNAPS_PYTHON_MODULE_NAME} MODULES ${VTK_LIBRARIES})
  target_compile_definitions(${VIENNAPS_PYTHON_MODULE_NAME} PRIVATE VIENNALS_VTK_MODULE_INIT)
endif()

# Put the built .so in build/<pkg> for convenience
set_target_properties(
  ${VIENNAPS_PYTHON_MODULE_NAME}
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${VIENNAPS_PYTHON_PACKAGE_NAME}"
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${VIENNAPS_PYTHON_PACKAGE_NAME}"
             ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${VIENNAPS_PYTHON_PACKAGE_NAME}")

if(VIENNAPS_USE_GPU)
  message(STATUS "[ViennaPS] Adding GPU bindings for 'viennaps' python package")
  add_dependencies(${VIENNAPS_PYTHON_MODULE_NAME} ${VIENNAPS_GPU_DEPENDENCIES})

  # Set PTX path override for runtime kernel loading from site-packages/<pkg>/ptx.
  # ABSOLUTE path enforced with dynamic module path. Only works on Linux based systems.
  # On Windows the dynamic path is relative to the python3xx.dll location which is not ideal.
  # So the PTX files are used from the build directory on Windows.
  if(NOT MSVC)
    target_compile_definitions(
      ${VIENNAPS_PYTHON_MODULE_NAME} PRIVATE VIENNACORE_KERNELS_PATH_OVERRIDE=ptx
                                             VIENNACORE_DYNAMIC_MODULE_PATH=1)
  endif()

  # Install PTX files
  install(DIRECTORY "${VIENNACORE_NVCC_PTX_DIR}/" DESTINATION "${VIENNAPS_PYTHON_PACKAGE_NAME}/ptx")
endif()

# Install into site-packages/<pkg>/  (RELATIVE path!)
install(
  TARGETS ${VIENNAPS_PYTHON_MODULE_NAME}
  LIBRARY DESTINATION ${VIENNAPS_PYTHON_PACKAGE_NAME}
  RUNTIME DESTINATION ${VIENNAPS_PYTHON_PACKAGE_NAME}
  ARCHIVE DESTINATION ${VIENNAPS_PYTHON_PACKAGE_NAME})

# Package files
install(FILES ${PROJECT_SOURCE_DIR}/__init__.py DESTINATION ${VIENNAPS_PYTHON_PACKAGE_NAME})
add_dependencies(${PROJECT_NAME} ${VIENNAPS_PYTHON_MODULE_NAME})

# --------------------------------------------------------------------------------------------------------
# Setup Lib-Folder
# --------------------------------------------------------------------------------------------------------

set(VIENNAPS_LIB_FOLDER "${VIENNAPS_LIBRARY_OUTPUT_DIR}/${VIENNAPS_PYTHON_PACKAGE_NAME}.libs")

viennacore_setup_embree_env(${VIENNAPS_PYTHON_MODULE_NAME} "${VIENNAPS_LIB_FOLDER}")
viennacore_setup_vtk_env(${VIENNAPS_PYTHON_MODULE_NAME} "${VIENNAPS_LIB_FOLDER}")
viennacore_setup_tbb_env(${VIENNAPS_PYTHON_MODULE_NAME} "${VIENNAPS_LIB_FOLDER}")

install(
  DIRECTORY ${VIENNAPS_LIB_FOLDER}
  DESTINATION .
  OPTIONAL)
