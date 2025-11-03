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

# Put the built .so in build/<pkg> for convenience
set_target_properties(
  ${VIENNAPS_PYTHON_MODULE_NAME}
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${VIENNAPS_PYTHON_PACKAGE_NAME}"
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${VIENNAPS_PYTHON_PACKAGE_NAME}"
             ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${VIENNAPS_PYTHON_PACKAGE_NAME}")

if(VIENNAPS_USE_GPU)
  message(STATUS "[ViennaPS] Adding GPU bindings for 'viennaps' python package")
  include("../cmake/generate_ptx.cmake")

  # Include pipelines of models
  generate_pipeline(${VIENNAPS_PYTHON_MODULE_NAME} generated_files
                    "${VIENNAPS_GPU_INCLUDE}/models/CallableWrapper.cu")

  # ViennaRay general pipelines
  generate_pipeline(${VIENNAPS_PYTHON_MODULE_NAME} generated_files
                    "${VIENNARAY_PIPELINE_DIR}/GeneralPipelineDisk.cu")
  generate_pipeline(${VIENNAPS_PYTHON_MODULE_NAME} generated_files
                    "${VIENNARAY_PIPELINE_DIR}/GeneralPipelineTriangle.cu")
  generate_pipeline(${VIENNAPS_PYTHON_MODULE_NAME} generated_files
                    "${VIENNARAY_PIPELINE_DIR}/GeneralPipelineLine.cu")

  # Generate CUDA kernels
  generate_kernel(${VIENNAPS_PYTHON_MODULE_NAME} generated_files ${VIENNARAY_CUDA_KERNELS})

  # Include ViennaPS GPU and OptiX headers
  target_sources(${VIENNAPS_PYTHON_MODULE_NAME} PRIVATE ${generated_files})
  target_include_directories(${VIENNAPS_PYTHON_MODULE_NAME} PRIVATE ${VIENNAPS_GPU_INCLUDE}
                                                                    ${VIENNARAY_GPU_INCLUDE})
  target_compile_definitions(${VIENNAPS_PYTHON_MODULE_NAME}
                             PRIVATE VIENNACORE_KERNELS_PATH_DEFINE=${VIENNACORE_PTX_DIR})

  # Install PTX files
  install(DIRECTORY "${VIENNACORE_PTX_DIR}/" DESTINATION "${VIENNAPS_PYTHON_PACKAGE_NAME}/ptx")
endif()

# Install into site-packages/<pkg>/  (RELATIVE path!)
install(
  TARGETS ${VIENNAPS_PYTHON_MODULE_NAME}
  LIBRARY DESTINATION ${VIENNAPS_PYTHON_PACKAGE_NAME}
  RUNTIME DESTINATION ${VIENNAPS_PYTHON_PACKAGE_NAME}
  ARCHIVE DESTINATION ${VIENNAPS_PYTHON_PACKAGE_NAME})

# Package files
install(FILES ${CMAKE_SOURCE_DIR}/python/__init__.py DESTINATION ${VIENNAPS_PYTHON_PACKAGE_NAME})

add_dependencies(${PROJECT_NAME} ${VIENNAPS_PYTHON_MODULE_NAME})

# --------------------------------------------------------------------------------------------------------
# Setup Lib-Folder
# --------------------------------------------------------------------------------------------------------

set(VIENNAPS_LIB_FOLDER "${VIENNAPS_LIBRARY_OUTPUT_DIR}/${VIENNAPS_PYTHON_PACKAGE_NAME}.libs")

# Not required for both targets, one will suffice
viennacore_setup_embree_env(${VIENNAPS_PYTHON_MODULE_NAME} "${VIENNAPS_LIB_FOLDER}")
viennacore_setup_vtk_env(${VIENNAPS_PYTHON_MODULE_NAME} "${VIENNAPS_LIB_FOLDER}")
viennacore_setup_tbb_env(${VIENNAPS_PYTHON_MODULE_NAME} "${VIENNAPS_LIB_FOLDER}")

install(
  DIRECTORY ${VIENNAPS_LIB_FOLDER}
  DESTINATION .
  OPTIONAL)
