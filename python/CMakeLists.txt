cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
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

if(APPLE AND VIENNAPS_PACKAGE_PYTHON)
  # Do not install viennaps.libs manually on macOS. These build-tree rpaths only
  # let delocate resolve VTK/Embree/TBB @rpath dependencies during wheel repair;
  # delocate then vendors the dylibs into viennaps/.dylibs and strips them.
  foreach(DEPENDENCY_TARGET VTK::CommonCore embree tbb)
    if(TARGET ${DEPENDENCY_TARGET})
      set_property(
        TARGET ${VIENNAPS_PYTHON_MODULE_NAME}
        APPEND
        PROPERTY INSTALL_RPATH "$<TARGET_FILE_DIR:${DEPENDENCY_TARGET}>")
    endif()
  endforeach()
endif()

# To prevent symbol clashes when multiple python scripts which use VTK modules are loaded, the modules have to be initialized here
if(VIENNALS_VTK_RENDERING)
  if(NOT TARGET VTK::CommonCore)
    if(NOT COMMAND viennaps_prepare_vtk)
      include("../cmake/vtk.cmake")
    endif()
    viennaps_prepare_vtk()
  endif()

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
if(WIN32)
  install(
    TARGETS ${VIENNAPS_PYTHON_MODULE_NAME} RUNTIME_DEPENDENCY_SET viennaps_python_runtime_deps
    LIBRARY DESTINATION ${VIENNAPS_PYTHON_PACKAGE_NAME}
    RUNTIME DESTINATION ${VIENNAPS_PYTHON_PACKAGE_NAME}
    ARCHIVE DESTINATION ${VIENNAPS_PYTHON_PACKAGE_NAME})
else()
  install(
    TARGETS ${VIENNAPS_PYTHON_MODULE_NAME}
    LIBRARY DESTINATION ${VIENNAPS_PYTHON_PACKAGE_NAME}
    RUNTIME DESTINATION ${VIENNAPS_PYTHON_PACKAGE_NAME}
    ARCHIVE DESTINATION ${VIENNAPS_PYTHON_PACKAGE_NAME})
endif()

# Package files
install(FILES ${PROJECT_SOURCE_DIR}/__init__.py DESTINATION ${VIENNAPS_PYTHON_PACKAGE_NAME})
add_dependencies(${PROJECT_NAME} ${VIENNAPS_PYTHON_MODULE_NAME})

# --------------------------------------------------------------------------------------------------------
# Setup Lib-Folder
# --------------------------------------------------------------------------------------------------------

set(VIENNAPS_LIB_FOLDER "${VIENNAPS_LIBRARY_OUTPUT_DIR}/${VIENNAPS_PYTHON_PACKAGE_NAME}.libs")

if(VIENNAPS_PACKAGE_PYTHON OR MSVC)
  # macOS wheels let delocate own dylib vendoring to avoid duplicate copies.
  # Other package builds still need this staged dependency folder.
  if(NOT APPLE AND NOT WIN32)
    viennacore_setup_embree_env(${VIENNAPS_PYTHON_MODULE_NAME} "${VIENNAPS_LIB_FOLDER}")
    viennacore_setup_vtk_env(${VIENNAPS_PYTHON_MODULE_NAME} "${VIENNAPS_LIB_FOLDER}")
    viennacore_setup_tbb_env(${VIENNAPS_PYTHON_MODULE_NAME} "${VIENNAPS_LIB_FOLDER}")
  endif()

  if(MSVC)
    # /openmp:llvm links against the LLVM OpenMP runtime, which is usually
    # available on GitHub runners but not on end-user Windows machines.
    set(CMAKE_INSTALL_OPENMP_LIBRARIES TRUE)
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
    include(InstallRequiredSystemLibraries)

    get_filename_component(_VIENNAPS_COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(_VIENNAPS_OPENMP_DLL_CANDIDATES)

    foreach(_VIENNAPS_SYSTEM_RUNTIME IN LISTS CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
      get_filename_component(_VIENNAPS_SYSTEM_RUNTIME_NAME "${_VIENNAPS_SYSTEM_RUNTIME}" NAME)
      if(_VIENNAPS_SYSTEM_RUNTIME_NAME MATCHES "^libomp.*\\.dll$")
        list(APPEND _VIENNAPS_OPENMP_DLL_CANDIDATES "${_VIENNAPS_SYSTEM_RUNTIME}")
      endif()
    endforeach()

    if(DEFINED ENV{VCINSTALLDIR})
      list(APPEND _VIENNAPS_OPENMP_DLL_CANDIDATES
           "$ENV{VCINSTALLDIR}/Tools/Llvm/bin/libomp140.x86_64.dll"
           "$ENV{VCINSTALLDIR}/Tools/Llvm/x64/bin/libomp140.x86_64.dll")
    endif()

    if(DEFINED ENV{VCToolsInstallDir})
      list(APPEND _VIENNAPS_OPENMP_DLL_CANDIDATES
           "$ENV{VCToolsInstallDir}/../../Llvm/bin/libomp140.x86_64.dll"
           "$ENV{VCToolsInstallDir}/../../Llvm/x64/bin/libomp140.x86_64.dll")
    endif()

    if(DEFINED ENV{llvmX64})
      list(APPEND _VIENNAPS_OPENMP_DLL_CANDIDATES "$ENV{llvmX64}/libomp140.x86_64.dll")
    endif()

    if(DEFINED ENV{VCToolsRedistDir})
      file(GLOB_RECURSE _VIENNAPS_OPENMP_REDIST_DLLS "$ENV{VCToolsRedistDir}/*libomp140.x86_64.dll")
      list(APPEND _VIENNAPS_OPENMP_DLL_CANDIDATES ${_VIENNAPS_OPENMP_REDIST_DLLS})
    endif()

    if(DEFINED ENV{VCINSTALLDIR})
      file(GLOB_RECURSE _VIENNAPS_OPENMP_VC_REDIST_DLLS
           "$ENV{VCINSTALLDIR}/Redist/MSVC/*libomp140.x86_64.dll")
      list(APPEND _VIENNAPS_OPENMP_DLL_CANDIDATES ${_VIENNAPS_OPENMP_VC_REDIST_DLLS})
    endif()

    find_file(
      _VIENNAPS_OPENMP_DLL_FROM_PATH
      NAMES libomp140.x86_64.dll
      PATHS ENV PATH
      NO_DEFAULT_PATH)
    if(_VIENNAPS_OPENMP_DLL_FROM_PATH)
      list(APPEND _VIENNAPS_OPENMP_DLL_CANDIDATES "${_VIENNAPS_OPENMP_DLL_FROM_PATH}")
    endif()

    list(
      APPEND
      _VIENNAPS_OPENMP_DLL_CANDIDATES
      "${_VIENNAPS_COMPILER_BIN_DIR}/libomp140.x86_64.dll"
      "${_VIENNAPS_COMPILER_BIN_DIR}/../libomp140.x86_64.dll"
      "${_VIENNAPS_COMPILER_BIN_DIR}/../../bin/libomp140.x86_64.dll"
      "${_VIENNAPS_COMPILER_BIN_DIR}/../../../../../Llvm/bin/libomp140.x86_64.dll"
      "${_VIENNAPS_COMPILER_BIN_DIR}/../../../../../Llvm/x64/bin/libomp140.x86_64.dll"
      "${_VIENNAPS_COMPILER_BIN_DIR}/../../../../../../Tools/Llvm/bin/libomp140.x86_64.dll"
      "${_VIENNAPS_COMPILER_BIN_DIR}/../../../../../../Tools/Llvm/x64/bin/libomp140.x86_64.dll")

    list(REMOVE_DUPLICATES _VIENNAPS_OPENMP_DLL_CANDIDATES)

    unset(_VIENNAPS_OPENMP_DLL)
    set(_VIENNAPS_OPENMP_DLL_SEARCH_LOG)
    foreach(_VIENNAPS_OPENMP_DLL_CANDIDATE IN LISTS _VIENNAPS_OPENMP_DLL_CANDIDATES)
      get_filename_component(_VIENNAPS_OPENMP_DLL_CANDIDATE_REAL
                             "${_VIENNAPS_OPENMP_DLL_CANDIDATE}" ABSOLUTE)
      list(APPEND _VIENNAPS_OPENMP_DLL_SEARCH_LOG "${_VIENNAPS_OPENMP_DLL_CANDIDATE_REAL}")
      if(EXISTS "${_VIENNAPS_OPENMP_DLL_CANDIDATE_REAL}")
        set(_VIENNAPS_OPENMP_DLL "${_VIENNAPS_OPENMP_DLL_CANDIDATE_REAL}")
        break()
      endif()
    endforeach()

    if(_VIENNAPS_OPENMP_DLL)
      message(STATUS "[ViennaPS] Bundling LLVM OpenMP runtime: ${_VIENNAPS_OPENMP_DLL}")
      add_custom_command(
        TARGET ${VIENNAPS_PYTHON_MODULE_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${VIENNAPS_LIB_FOLDER}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_VIENNAPS_OPENMP_DLL}"
                "${VIENNAPS_LIB_FOLDER}/"
        VERBATIM)

      if(VIENNAPS_PACKAGE_PYTHON)
        install(PROGRAMS "${_VIENNAPS_OPENMP_DLL}" DESTINATION ${VIENNAPS_PYTHON_PACKAGE_NAME}.libs)
      endif()
    elseif(VIENNAPS_PACKAGE_PYTHON)
      string(REPLACE ";" "\n  " _VIENNAPS_OPENMP_DLL_SEARCH_LOG_TEXT
                     "${_VIENNAPS_OPENMP_DLL_SEARCH_LOG}")
      message(
        FATAL_ERROR
          "[ViennaPS] Could not locate libomp140.x86_64.dll for a Windows wheel build. "
          "The produced wheel would fail to import on machines without the LLVM OpenMP runtime. "
          "Searched:\n  ${_VIENNAPS_OPENMP_DLL_SEARCH_LOG_TEXT}")
    else()
      message(
        WARNING "[ViennaPS] Could not locate libomp140.x86_64.dll. Windows wheels built "
                "with /openmp:llvm may fail to import on machines without the LLVM OpenMP runtime.")
    endif()
  endif()
endif()

if(WIN32)
  set(VIENNAPS_RUNTIME_DEPENDENCY_DIRS)
  foreach(DEPENDENCY_TARGET VTK::CommonCore embree tbb)
    if(TARGET ${DEPENDENCY_TARGET})
      list(APPEND VIENNAPS_RUNTIME_DEPENDENCY_DIRS "$<TARGET_FILE_DIR:${DEPENDENCY_TARGET}>")
    endif()
  endforeach()

  install(
    RUNTIME_DEPENDENCY_SET
    viennaps_python_runtime_deps
    DESTINATION
    ${VIENNAPS_PYTHON_PACKAGE_NAME}.libs
    DIRECTORIES
    ${VIENNAPS_RUNTIME_DEPENDENCY_DIRS}
    PRE_EXCLUDE_REGEXES
    "api-ms-"
    "ext-ms-"
    "^python[0-9]+[a-z]*\\.dll$"
    POST_EXCLUDE_REGEXES
    "[/\\\\][Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\\\][Ss][Yy][Ss][Tt][Ee][Mm]32[/\\\\]"
    "[/\\\\][Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\\\][Ss][Yy][Ss][Ww][Oo][Ww]64[/\\\\]"
    "[/\\\\][Ww][Ii][Nn][Ss][Xx][Ss][/\\\\]"
    "[/\\\\]Python[0-9]+[/\\\\]"
    "[/\\\\]python[0-9]+[a-z]*\\.dll$"
    "[/\\\\]vcruntime[0-9]+.*\\.dll$"
    "[/\\\\]msvcp[0-9]+.*\\.dll$")
else()
  install(
    DIRECTORY ${VIENNAPS_LIB_FOLDER}
    DESTINATION .
    OPTIONAL)
endif()
