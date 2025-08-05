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
  VERSION 2.12.0
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
# Setup Bindings
# --------------------------------------------------------------------------------------------------------

set(LSNAME "viennals2d")
viennacore_setup_binding(viennaps2d ${VIENNAPS_LIBRARY_OUTPUT_DIR})

set(LSNAME "viennals3d")
viennacore_setup_binding(viennaps3d ${VIENNAPS_LIBRARY_OUTPUT_DIR})

if(VIENNAPS_USE_GPU)
  message(STATUS "[ViennaPS] Adding GPU bindings for 'viennaps3d'")
  include("../cmake/generate_ptx.cmake")

  # Include pipeline files
  set(generated_files) # Empty list variable
  file(GLOB files "${VIENNAPS_PIPELINE_DIR}/*.cu")
  foreach(file ${files})
    get_filename_component(filename ${file} NAME)
    generate_pipeline(viennaps3d generated_files "${VIENNAPS_PIPELINE_DIR}/${filename}")
  endforeach()
  # Generate CUDA kernels
  generate_kernel(viennaps3d generated_files ${VIENNARAY_CUDA_KERNELS})

  # Include ViennaPS GPU and OptiX headers
  target_sources(viennaps3d PRIVATE ${generated_files})
  target_include_directories(viennaps3d PRIVATE ${VIENNAPS_GPU_INCLUDE} ${VIENNARAY_GPU_INCLUDE}
                                                ${OptiX_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
  target_link_libraries(viennaps3d PRIVATE ViennaPS ${VIENNACORE_GPU_LIBS})
  target_compile_definitions(viennaps3d PRIVATE VIENNAPS_USE_GPU)
  target_compile_definitions(viennaps3d
                             PRIVATE VIENNACORE_KERNELS_PATH_DEFINE=${VIENNACORE_PTX_DIR})

  # We have to include the CUDA headers for the 2D version as well because ViennaCore uses them in the Logger
  target_include_directories(viennaps2d PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(viennaps2d PRIVATE ViennaPS ${VIENNACORE_GPU_LIBS})

  add_custom_command(
    TARGET viennaps3d
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${VIENNACORE_PTX_DIR}"
            "${CMAKE_SOURCE_DIR}/python/viennaps3d/ptx"
    COMMENT "Copying .ptx and .optixir files to Python package directory")
else()
  target_link_libraries(viennaps3d PRIVATE ViennaPS)
  target_link_libraries(viennaps2d PRIVATE ViennaPS)
endif()

target_compile_definitions(viennaps2d PRIVATE -DVIENNAPS_PYTHON_DIMENSION=2
                                              -DVIENNAPS_MODULE_NAME=viennaps2d)
target_compile_definitions(viennaps3d PRIVATE -DVIENNAPS_PYTHON_DIMENSION=3
                                              -DVIENNAPS_MODULE_NAME=viennaps3d)

# --------------------------------------------------------------------------------------------------------
# Setup Lib-Folder
# --------------------------------------------------------------------------------------------------------

set(VIENNAPS_LIB_FOLDER "${VIENNAPS_LIBRARY_OUTPUT_DIR}/viennaps.libs")

# Not required for both targets, one will suffice
viennacore_setup_embree_env(viennaps3d "${VIENNAPS_LIB_FOLDER}")
viennacore_setup_vtk_env(viennaps3d "${VIENNAPS_LIB_FOLDER}")
viennacore_setup_tbb_env(viennaps3d "${VIENNAPS_LIB_FOLDER}")

install(
  DIRECTORY "${VIENNAPS_LIB_FOLDER}"
  DESTINATION .
  OPTIONAL)
