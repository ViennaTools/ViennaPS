function(viennaps_patch_vtk_msvc_stdext VTK_SOURCE_DIR)
  if(NOT MSVC)
    return()
  endif()

  set(_vtk_fmt_header "${VTK_SOURCE_DIR}/ThirdParty/diy2/vtkdiy2/include/vtkdiy2/fmt/format.h")

  if(NOT EXISTS "${_vtk_fmt_header}")
    message(
      WARNING
        "[ViennaPS] Could not find VTK diy2/fmt header for MSVC stdext patch: ${_vtk_fmt_header}")
    return()
  endif()

  file(READ "${_vtk_fmt_header}" _vtk_fmt_contents)

  set(_patched_guard "#if defined(_SECURE_SCL) && (!defined(_MSC_VER) || _MSC_VER < 1951)")
  string(FIND "${_vtk_fmt_contents}" "${_patched_guard}" _already_patched)

  if(NOT _already_patched EQUAL -1)
    message(STATUS "[ViennaPS] VTK MSVC stdext patch already applied")
    return()
  endif()

  set(_old_guard "#ifdef _SECURE_SCL")
  string(FIND "${_vtk_fmt_contents}" "${_old_guard}" _old_guard_pos)

  if(_old_guard_pos EQUAL -1)
    message(
      WARNING
        "[ViennaPS] VTK MSVC stdext patch was not applied; expected guard not found in ${_vtk_fmt_header}"
    )
    return()
  endif()

  string(REPLACE "${_old_guard}" "${_patched_guard}" _vtk_fmt_contents "${_vtk_fmt_contents}")
  file(WRITE "${_vtk_fmt_header}" "${_vtk_fmt_contents}")

  message(STATUS "[ViennaPS] Applied VTK MSVC stdext patch")
endfunction()

function(viennaps_patch_vtk_openmp_nested VTK_SOURCE_DIR)
  file(GLOB_RECURSE _vtk_smp_openmp_sources "${VTK_SOURCE_DIR}/Common/Core/SMP/OpenMP/*.cxx"
       "${VTK_SOURCE_DIR}/Common/Core/SMP/OpenMP/*.txx"
       "${VTK_SOURCE_DIR}/Common/Core/SMP/OpenMP/*.h")

  if(NOT _vtk_smp_openmp_sources)
    message(WARNING "[ViennaPS] Could not find VTK OpenMP SMP sources for omp_set_nested patch")
    return()
  endif()

  set(_vtk_smp_openmp_patch_count 0)

  foreach(_vtk_smp_openmp IN LISTS _vtk_smp_openmp_sources)
    file(READ "${_vtk_smp_openmp}" _vtk_smp_contents)

    string(FIND "${_vtk_smp_contents}" "omp_set_nested(" _has_set_nested)
    string(FIND "${_vtk_smp_contents}" "omp_get_nested()" _has_get_nested)

    if(_has_set_nested EQUAL -1 AND _has_get_nested EQUAL -1)
      continue()
    endif()

    if(NOT _has_set_nested EQUAL -1)
      string(REGEX REPLACE [[omp_set_nested\(([^)]*)\);]] [[/* VIENNAPS_PATCH_OMP_SET_NESTED */
    omp_set_max_active_levels((\1) ? 1024 : 1);]] _vtk_smp_contents "${_vtk_smp_contents}")
    endif()

    if(NOT _has_get_nested EQUAL -1)
      string(
        REGEX
        REPLACE "omp_get_nested\\(\\)"
                "/* VIENNAPS_PATCH_OMP_GET_NESTED */ (omp_get_max_active_levels() > 1)"
                _vtk_smp_contents "${_vtk_smp_contents}")
    endif()

    string(FIND "${_vtk_smp_contents}" "omp_set_nested(" _still_has_set_nested)
    if(NOT _still_has_set_nested EQUAL -1)
      message(
        FATAL_ERROR
          "[ViennaPS] VTK OpenMP nested-parallelism patch did not remove omp_set_nested from ${_vtk_smp_openmp}"
      )
    endif()

    file(WRITE "${_vtk_smp_openmp}" "${_vtk_smp_contents}")
    math(EXPR _vtk_smp_openmp_patch_count "${_vtk_smp_openmp_patch_count} + 1")
    message(STATUS "[ViennaPS] Patched VTK OpenMP nested-parallelism source: ${_vtk_smp_openmp}")
  endforeach()

  if(_vtk_smp_openmp_patch_count EQUAL 0)
    message(STATUS "[ViennaPS] VTK OpenMP nested-parallelism patch not needed")
  else()
    message(STATUS "[ViennaPS] Applied VTK OpenMP nested-parallelism patch")
  endif()
endfunction()

function(viennaps_prepare_vtk)
  if(TARGET VTK::CommonCore)
    message(STATUS "[ViennaPS] Reusing previously configured VTK targets")
  else()
    enable_language(C)

    set(VTK_OPTIONS
        "BUILD_SHARED_LIBS OFF"
        "VTK_INSTALL_SDK OFF"
        "VTK_BUILD_TESTING OFF"
        "BUILD_TESTING OFF"
        "VTK_BUILD_ALL_MODULES OFF"
        "VTK_ENABLE_REMOTE_MODULES OFF"
        "VTK_LEGACY_REMOVE ON"
        "VTK_SMP_IMPLEMENTATION_TYPE OpenMP"
        "VTK_SMP_ENABLE_STDTHREAD OFF"
        "VTK_SMP_ENABLE_OPENMP ON"
        "VTK_SMP_ENABLE_TBB OFF"
        "VTK_GROUP_ENABLE_Rendering DONT_WANT"
        "VTK_GROUP_ENABLE_Imaging DONT_WANT"
        "VTK_GROUP_ENABLE_Views DONT_WANT"
        "VTK_GROUP_ENABLE_Web DONT_WANT"
        "VTK_GROUP_ENABLE_Qt NO"
        "VTK_GROUP_ENABLE_MPI NO"
        "VTK_ENABLE_WRAPPING NO"
        "VTK_MODULE_ENABLE_VTK_libproj NO"
        "VTK_MODULE_ENABLE_VTK_CommonExecutionModel YES"
        "VTK_MODULE_ENABLE_VTK_CommonMisc YES"
        "VTK_MODULE_ENABLE_VTK_CommonSystem YES"
        "VTK_MODULE_ENABLE_VTK_CommonMath YES"
        "VTK_MODULE_ENABLE_VTK_CommonCore YES"
        "VTK_MODULE_ENABLE_VTK_CommonTransforms YES"
        "VTK_MODULE_ENABLE_VTK_CommonComputationalGeometry YES"
        "VTK_MODULE_ENABLE_VTK_IOCore YES"
        "VTK_MODULE_ENABLE_VTK_IOXMLParser YES"
        "VTK_MODULE_ENABLE_VTK_IOXML YES"
        "VTK_MODULE_ENABLE_VTK_FiltersCore YES"
        "VTK_MODULE_ENABLE_VTK_FiltersGeneral YES"
        "VTK_MODULE_ENABLE_VTK_FiltersGeometry YES"
        "VTK_GROUP_ENABLE_Parallel DONT_WANT"
        "VTK_MODULE_ENABLE_VTK_ParallelDIY DONT_WANT"
        "VTK_MODULE_ENABLE_VTK_ParallelCore DONT_WANT")

    if(VIENNAPS_VTK_RENDERING)
      list(
        APPEND
        VTK_OPTIONS
        "VTK_MODULE_ENABLE_VTK_RenderingCore YES"
        "VTK_MODULE_ENABLE_VTK_RenderingOpenGL2 YES"
        "VTK_MODULE_ENABLE_VTK_RenderingUI YES"
        "VTK_MODULE_ENABLE_VTK_InteractionStyle YES"
        "VTK_MODULE_ENABLE_VTK_RenderingFreeType YES"
        "VTK_MODULE_ENABLE_VTK_IOImage YES"
        "VTK_MODULE_ENABLE_VTK_RenderingAnnotation YES")
    endif()

    if(APPLE)
      list(APPEND VTK_OPTIONS "VTK_MODULE_USE_EXTERNAL_VTK_png ON"
           "VTK_MODULE_USE_EXTERNAL_VTK_zlib ON")
    endif()

    set(VTK_SMP_IMPLEMENTATION_TYPE
        "OpenMP"
        CACHE STRING "" FORCE)
    set(VTK_SMP_ENABLE_STDTHREAD
        OFF
        CACHE BOOL "" FORCE)
    set(VTK_SMP_ENABLE_OPENMP
        ON
        CACHE BOOL "" FORCE)
    set(VTK_SMP_ENABLE_TBB
        OFF
        CACHE BOOL "" FORCE)

    if(NOT VIENNAPS_PACKAGE_PYTHON)
      find_package(VTK 9.0.0 QUIET)
    endif()

    if(VTK_FOUND)
      message(STATUS "[ViennaPS] Using system VTK ${VTK_VERSION}")
      set(VTK_FOUND TRUE PARENT_SCOPE)
    else()
      message(STATUS "[ViennaPS] Using bundled VTK v9.3.1")

      foreach(_vtk_option IN LISTS VTK_OPTIONS)
        if(_vtk_option MATCHES "^([^ ]+) +(.*)$")
          set("${CMAKE_MATCH_1}"
              "${CMAKE_MATCH_2}"
              CACHE STRING "" FORCE)
        else()
          message(FATAL_ERROR "[ViennaPS] Invalid VTK option: ${_vtk_option}")
        endif()
      endforeach()

      CPMAddPackage(
        NAME VTK
        GIT_TAG v9.3.1
        GIT_REPOSITORY "https://gitlab.kitware.com/vtk/vtk"
        DOWNLOAD_ONLY YES)

      if(VTK_ADDED)
        viennaps_patch_vtk_msvc_stdext("${VTK_SOURCE_DIR}")
        viennaps_patch_vtk_openmp_nested("${VTK_SOURCE_DIR}")

        if(VIENNAPS_BUILD_PYTHON)
          add_subdirectory("${VTK_SOURCE_DIR}" "${VTK_BINARY_DIR}" EXCLUDE_FROM_ALL)
        else()
          add_subdirectory("${VTK_SOURCE_DIR}" "${VTK_BINARY_DIR}")
        endif()
      endif()
    endif()
  endif()

  message(STATUS "[ViennaPS] VTK_SMP_IMPLEMENTATION_TYPE=${VTK_SMP_IMPLEMENTATION_TYPE}")
  message(STATUS "[ViennaPS] VTK_SMP_ENABLE_OPENMP=${VTK_SMP_ENABLE_OPENMP}")
  message(STATUS "[ViennaPS] VTK_SMP_ENABLE_STDTHREAD=${VTK_SMP_ENABLE_STDTHREAD}")
  message(STATUS "[ViennaPS] VTK_SMP_ENABLE_TBB=${VTK_SMP_ENABLE_TBB}")

  set(VTK_LIBRARIES
      VTK::CommonExecutionModel
      VTK::CommonMisc
      VTK::CommonSystem
      VTK::CommonMath
      VTK::CommonCore
      VTK::CommonTransforms
      VTK::CommonComputationalGeometry
      VTK::IOCore
      VTK::IOXMLParser
      VTK::IOXML
      VTK::FiltersCore
      VTK::FiltersGeneral
      VTK::FiltersGeometry
      CACHE INTERNAL "VTK Libraries")

  if(VIENNAPS_VTK_RENDERING)
    list(
      APPEND
      VTK_LIBRARIES
      VTK::RenderingCore
      VTK::RenderingOpenGL2
      VTK::InteractionStyle
      VTK::RenderingUI
      VTK::RenderingFreeType
      VTK::RenderingAnnotation
      VTK::IOImage)
    set(VTK_LIBRARIES
        "${VTK_LIBRARIES}"
        CACHE INTERNAL "VTK Libraries" FORCE)
  endif()
endfunction()
