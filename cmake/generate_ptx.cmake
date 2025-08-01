function(generate_pipeline target_name generated_files_output)

  # Separate the sources from the CMake and CUDA options fed to the macro.  This code
  # comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.
  cuda_get_sources_and_options(cu_optix_source_files cmake_options options ${ARGN})

  # Add the path to the OptiX headers to our include paths.
  cuda_include_directories(${OptiX_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

  # Include ViennaRay headers which are used in pipelines
  cuda_include_directories(${VIENNARAY_GPU_INCLUDE} ${VIENNAPS_GPU_INCLUDE})
  cuda_include_directories(${ViennaPS_SOURCE_DIR}/include/viennaps)
  cuda_include_directories(${ViennaCore_SOURCE_DIR}/include/viennacore)
  add_compile_definitions(VIENNACORE_COMPILE_GPU)

  # Generate OptiX IR files if enabled
  if(VIENNAPS_GENERATE_OPTIXIR)
    cuda_wrap_srcs(
      ${target_name}
      OPTIXIR
      generated_optixir_files
      ${cu_optix_source_files}
      ${cmake_options}
      OPTIONS
      ${options})
    list(APPEND generated_files_local ${generated_optixir_files})
  endif()

  # Generate PTX files if enabled
  if(VIENNAPS_GENERATE_PTX)
    cuda_wrap_srcs(
      ${target_name}
      PTX
      generated_ptx_files
      ${cu_optix_source_files}
      ${cmake_options}
      OPTIONS
      ${options})
    list(APPEND generated_files_local ${generated_ptx_files})
  endif()

  set(${generated_files_output}
      "${generated_files_local};${${generated_files_output}}"
      PARENT_SCOPE)
endfunction()

function(generate_kernel target_name generated_files_output)

  # Separate the sources from the CMake and CUDA options fed to the macro.  This code
  # comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.
  cuda_get_sources_and_options(cu_source_files cmake_options options ${ARGN})

  cuda_include_directories(${OptiX_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
  cuda_include_directories(${ViennaCore_SOURCE_DIR}/include/viennacore)
  cuda_include_directories(${VIENNARAY_GPU_INCLUDE} ${VIENNAPS_GPU_INCLUDE})
  add_compile_definitions(VIENNACORE_COMPILE_GPU)

  cuda_wrap_srcs(
    ${target_name}
    PTX
    generated_ptx_files
    ${cu_source_files}
    ${cmake_options}
    OPTIONS
    ${options})

  set(${generated_files_output}
      "${generated_ptx_files};${${generated_files_output}}"
      PARENT_SCOPE)
endfunction()

# In CMake, functions have their own scope, whereas macros use the scope of the caller.
function(add_GPU_executable target_name_base target_name_var)
  set(target_name ${target_name_base})
  set(${target_name_var}
      ${target_name}
      PARENT_SCOPE)

  # Separate the sources from the CMake and CUDA options fed to the macro.  This code
  # comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.  We are copying the
  # code here, so that we can use our own name for the target.  target_name is used in the
  # creation of the output file names, and we want this to be unique for each target in
  # the SDK.
  cuda_get_sources_and_options(source_files cmake_options options ${ARGN})

  # Isolate OBJ target files. NVCC should only process these files and leave PTX targets for NVRTC
  set(cu_obj_source_files)
  set(cu_optix_source_files)
  foreach(file ${source_files})
    get_filename_component(_file_extension ${file} EXT)
    if(${_file_extension} MATCHES "cu")
      list(APPEND cu_optix_source_files ${file})
    endif()
  endforeach()

  # Add the path to the OptiX headers to our include paths.
  cuda_include_directories(${OptiX_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

  # Include ViennaRay headers which are used in pipelines
  cuda_include_directories(${VIENNARAY_GPU_INCLUDE} ${VIENNAPS_GPU_INCLUDE})
  cuda_include_directories(${ViennaPS_SOURCE_DIR}/include/viennaps)
  cuda_include_directories(${ViennaCore_SOURCE_DIR}/include/viennacore)
  add_compile_definitions(VIENNACORE_COMPILE_GPU)

  # Create CUDA kernels
  cuda_wrap_srcs(
    ${target_name}
    PTX
    generated_files
    ${VIENNARAY_CUDA_KERNELS}
    ${VIENNAPS_CUDA_KERNELS}
    ${cmake_options}
    OPTIONS
    ${options})

  # Create the rules to build the PTX and/or OPTIX files.
  if(VIENNAPS_GENERATE_OPTIXIR)
    cuda_wrap_srcs(
      ${target_name}
      OPTIXIR
      generated_optixir_files
      ${cu_optix_source_files}
      ${cmake_options}
      OPTIONS
      ${options})
    list(APPEND generated_files ${generated_optixir_files})
  endif()
  if(VIENNAPS_GENERATE_PTX)
    cuda_wrap_srcs(
      ${target_name}
      PTX
      generated_ptx_files
      ${cu_optix_source_files}
      ${cmake_options}
      OPTIONS
      ${options})
    list(APPEND generated_files ${generated_ptx_files})
  endif()

  # Here is where we create the rule to make the executable.  We define a target name and
  # list all the source files used to create the target.  In addition we also pass along
  # the cmake_options parsed out of the arguments.
  message(STATUS "Adding target: ${target_name}")
  add_executable(${target_name} ${source_files} ${generated_files} ${cmake_options})
  target_include_directories(${target_name} PRIVATE ${OptiX_INCLUDE_DIR} ${VIENNARAY_GPU_INCLUDE}
                                                    ${VIENNAPS_GPU_INCLUDE} ${CUDA_INCLUDE_DIRS})
  target_link_libraries(${target_name} PRIVATE ViennaPS ${VIENNACORE_GPU_LIBS})
  target_compile_definitions(${target_name}
                             PRIVATE VIENNACORE_KERNELS_PATH_DEFINE=${VIENNACORE_PTX_DIR})
endfunction()
