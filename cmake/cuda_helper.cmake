# Helper macros to configure CUDA and OptiX

macro(add_cuda_flag_config config flag)
  string(TOUPPER "${config}" config)
  list(FIND CUDA_NVCC_FLAGS${config} ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS${config} ${flag})
    set(CUDA_NVCC_FLAGS${config}
        ${CUDA_NVCC_FLAGS${config}}
        CACHE STRING ${CUDA_NVCC_FLAGS_DESCRIPTION} FORCE)
  endif()
endmacro()

macro(add_cuda_flag flag)
  add_cuda_flag_config("" ${flag})
endmacro()

macro(generate_pipeline target_name generated_files)

  # Separate the sources from the CMake and CUDA options fed to the macro.  This code
  # comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.
  cuda_get_sources_and_options(cu_optix_source_files cmake_options options ${ARGN})

  # Add the path to the OptiX headers to our include paths.
  include_directories(${OptiX_INCLUDE})

  # Include ViennaPS headers which are used in pipelines
  include_directories(${VIENNAPS_GPU_INCLUDE_DIRS})
  include_directories(${ViennaCore_SOURCE_DIR}/include/viennacore) # needed for Context
  include_directories(${CMAKE_SOURCE_DIR}/include/viennaps/models)
  add_compile_definitions(VIENNACORE_COMPILE_GPU)

  # Generate OptiX IR files if enabled
  if(VIENNAPS_INPUT_GENERATE_OPTIXIR)
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
  if(VIENNAPS_INPUT_GENERATE_PTX)
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

  list(APPEND ${generated_files} ${generated_files_local})
endmacro()

macro(generate_kernel target_name generated_files)

  # Separate the sources from the CMake and CUDA options fed to the macro.  This code
  # comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.
  cuda_get_sources_and_options(cu_source_files cmake_options options ${ARGN})

  cuda_wrap_srcs(
    ${target_name}
    PTX
    generated_ptx_files
    ${cu_source_files}
    ${cmake_options}
    OPTIONS
    ${options})

  list(APPEND ${generated_files} ${generated_ptx_files})
endmacro()
