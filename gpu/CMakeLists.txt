#
# SPDX-FileCopyrightText: Copyright (c) 2019 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

project(ViennaPS-GPU)

# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake (i.e. FindCUDA and FindOptiX).
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL TRUE)

# Find at least a 5.0 version of CUDA.
find_package(CUDA 5.0 REQUIRED)

# Find OptiX
if("${is_cached}" STREQUAL "" AND DEFINED OptiX_INSTALL_DIR)
  # Required for windows to convert backslashes to forward slashes
  file(TO_CMAKE_PATH "${OptiX_INSTALL_DIR}" OptiX_INSTALL_DIR)
  set(OptiX_INSTALL_DIR
      "${OptiX_INSTALL_DIR}"
      CACHE PATH "Path to OptiX installation" FORCE)
else()
  set(OptiX_INSTALL_DIR
      $ENV{OptiX_INSTALL_DIR}
      CACHE PATH "Path to OptiX installation.")
endif()

# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)

if(CUDA_VERSION VERSION_LESS 11.7)
  if(VIENNAPS_INPUT_ENABLE_OPTIXIR_SUPPORT)
    message(
      SEND_ERROR
        "VIENNAPS_INPUT_ENABLE_OPTIXIR_SUPPORT is not supported in CUDA versions less than 11.7")
  else()
    option(VIENNAPS_INPUT_ENABLE_OPTIXIR_SUPPORT
           "Enable support for generating OptiX-IR targeted input files" OFF)
  endif()
else()
  option(VIENNAPS_INPUT_ENABLE_OPTIXIR_SUPPORT
         "Enable support for generating OptiX-IR targeted input files" ON)
endif()

if(VIENNAPS_INPUT_ENABLE_OPTIXIR_SUPPORT)
  option(VIENNAPS_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" ON)
  option(VIENNAPS_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" OFF)
else()
  option(VIENNAPS_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" OFF)
  option(VIENNAPS_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" ON)
endif()

# Add the path to the OptiX headers to our include paths.
include_directories("${OptiX_INCLUDE}")

# Also ViennaPS headers
set(VIENNAPS_GPU_INCLUDE_DIRS #
    ${CMAKE_SOURCE_DIR}/app
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/geometry
    ${PROJECT_SOURCE_DIR}/process
    ${PROJECT_SOURCE_DIR}/rayTracing
    ${PROJECT_SOURCE_DIR}/utils
    ${PROJECT_SOURCE_DIR}/app
    ${PROJECT_SOURCE_DIR}/models
    ${ViennaCore_SOURCE_DIR}/include/viennacore)
include_directories(${VIENNAPS_GPU_INCLUDE_DIRS})

####

set(VIENNAPS_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx")
set(CUDA_GENERATED_OUTPUT_DIR ${VIENNAPS_PTX_DIR})

if(WIN32)
  string(REPLACE "/" "\\\\" VIENNAPS_PTX_DIR ${VIENNAPS_PTX_DIR})
else(WIN32)
  if(USING_GNU_C AND NOT APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DM_PI=3.14159265358979323846")
  endif()
endif(WIN32)

# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

set(CUDA_MIN_SM_TARGET
    sm_50
    CACHE STRING "Minimum CUDA SM architecture to use for compilation.")

macro(optix_add_cuda_flag_config config flag)
  string(TOUPPER "${config}" config)
  list(FIND CUDA_NVCC_FLAGS${config} ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS${config} ${flag})
    set(CUDA_NVCC_FLAGS${config}
        ${CUDA_NVCC_FLAGS${config}}
        CACHE STRING ${CUDA_NVCC_FLAGS_DESCRIPTION} FORCE)
  endif()
endmacro()

macro(optix_add_cuda_flag flag)
  optix_add_cuda_flag_config("" ${flag})
endmacro()

# Add some useful default arguments to the NVCC and NVRTC flags.  This is an example of
# how we use PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE
# and following block of code will not be executed leaving you free to edit the values
# as much as you wish from the GUI or from ccmake.
if(NOT PASSED_FIRST_CONFIGURE)
  message(STATUS "Setting default NVCC flags for first time configuration.")

  set(CUDA_NVCC_FLAGS_DESCRIPTION "Semi-colon delimit multiple arguments.")
  string(REPLACE "sm_" "compute_" CUDA_MIN_SM_COMPUTE_TARGET ${CUDA_MIN_SM_TARGET})

  list(FIND CUDA_NVCC_FLAGS "-arch" index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS -arch ${CUDA_MIN_SM_TARGET})
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}
        CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  endif()

  optix_add_cuda_flag("--use_fast_math")
  optix_add_cuda_flag("-lineinfo")
  optix_add_cuda_flag("--expt-relaxed-constexpr")
  optix_add_cuda_flag("--generate-line-info")

  # CMAKE_CONFIGURATION_TYPES is only defined for multi-config build systems like
  # MSVC and Ninja, but we need to generate flags for each configuration
  # regardless.
  if(DEFINED CMAKE_CONFIGURATION_TYPES)
    set(OPTIX_CONFIGURATION_TYPES ${CMAKE_CONFIGURATION_TYPES})
  else()
    set(OPTIX_CONFIGURATION_TYPES "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
  endif()

  foreach(config ${OPTIX_CONFIGURATION_TYPES})
    if(${config} STREQUAL "Debug")
      optix_add_cuda_flag_config(_${config} "-G")
      optix_add_cuda_flag_config(_${config} "-O0")
    endif()
  endforeach()

  if(CUDA_VERSION VERSION_LESS "3.0")
    optix_add_cuda_flag("--keep")
  endif()

  # Some CUDA 11.x toolkits erroneously complain about sm_50 being deprecated
  if(CUDA_VERSION VERSION_GREATER "11.0")
    optix_add_cuda_flag("-Wno-deprecated-gpu-targets")
  endif()

  if(CUDA_USE_LOCAL_ENV)
    optix_add_cuda_flag("--use-local-env")
  endif()
endif()

# message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

set(VIENNAPS_CUDA_KERNELS ${PROJECT_SOURCE_DIR}/deviceKernels/normKernels.cu)

# Mark all pipeline files as Optix files (DOES NOT WORK)
# file(GLOB OPTIX_FILES "pipelines/*.cu")
# foreach(FILE ${OPTIX_FILES})
#   set_source_files_properties(${FILE} PROPERTIES LABELS "OPTIX")
# endforeach()

# In CMake, functions have their own scope, whereas macros use the scope of the caller.
macro(add_GPU_executable target_name_base target_name_var)

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

  add_compile_definitions(VIENNACORE_COMPILE_GPU)
  cuda_wrap_srcs(
    ${target_name}
    PTX
    generated_files
    ${VIENNAPS_CUDA_KERNELS}
    ${cmake_options}
    OPTIONS
    ${options})



  # Create the rules to build the PTX and/or OPTIX files.
  if(VIENNAPS_INPUT_GENERATE_OPTIXIR)
    cuda_wrap_srcs(
      ${target_name}
      OPTIXIR
      generated_files2
      ${cu_optix_source_files}
      ${cmake_options}
      OPTIONS
      ${options})
    list(APPEND generated_files ${generated_files2})
  endif()
  if(VIENNAPS_INPUT_GENERATE_PTX)
    cuda_wrap_srcs(
      ${target_name}
      PTX
      generated_files3
      ${cu_optix_source_files}
      ${cmake_options}
      OPTIONS
      ${options})
    list(APPEND generated_files ${generated_files3})
  endif()

  # Here is where we create the rule to make the executable.  We define a target name and
  # list all the source files used to create the target.  In addition we also pass along
  # the cmake_options parsed out of the arguments.
  add_executable(${target_name} ${source_files} ${generated_files} ${cmake_options})
  target_link_libraries(${target_name} ViennaPS ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})

  message(STATUS "Adding target: ${target_name}")

  target_compile_definitions(${target_name} PRIVATE VIENNAPS_KERNELS_PATH_DEFINE=${VIENNAPS_PTX_DIR})

  if(UNIX AND NOT APPLE)
    # Force using RPATH instead of RUNPATH on Debian
    target_link_libraries(${target_name} "-Wl,--disable-new-dtags")
  endif()

  if(USING_GNU_CXX)
    # Explicitly link against math library (C samples don't do that by default)
    target_link_libraries(${target_name} m)
  endif()
endmacro()

add_subdirectory(examples)
add_subdirectory(benchmark)
add_subdirectory(tests)

# add_custom_target(buildGPUApplication)

# include_directories(${VIENNAPS_GPU_INCLUDE_DIRS})
# add_compile_definitions(VIENNACORE_COMPILE_GPU)
# file(GLOB kernels "${PROJECT_SOURCE_DIR}/deviceKernels/*.cu")
# foreach(kernel ${kernels})

#   cuda_compile_ptx(
#     cuda_ptx_files
#     ${kernel}
#     OPTIONS
#     --generate-line-info
#     -use_fast_math
#     --keep
#     -gencode=arch=compute_80,code=sm_80
#     --expt-relaxed-constexpr)
#   get_filename_component(kernelName ${kernel} NAME_WE)

#   add_custom_command(
#     OUTPUT ${CMAKE_BINARY_DIR}/${kernelName}.ptx
#     COMMAND ${CMAKE_COMMAND} -E rename ${cuda_ptx_files} ${CMAKE_BINARY_DIR}/${kernelName}.ptx
#     DEPENDS ${cuda_ptx_files})

#   add_custom_target(
#     ${kernelName} ALL
#     DEPENDS ${CMAKE_BINARY_DIR}/${kernelName}.ptx ${kernel}
#     SOURCES ${kernel})

#   add_dependencies(buildGPUApplication ${kernelName})
# endforeach()

# cuda_compile_and_embed(embedded_SF6O2_pipeline ${PROJECT_SOURCE_DIR}/pipelines/SF6O2Pipeline.cu)
# cuda_compile_and_embed(embedded_Fluorocarbon_pipeline
#                        ${PROJECT_SOURCE_DIR}/pipelines/FluorocarbonPipeline.cu)

# cuda_compile_and_embed(embedded_SingleParticle_pipeline
#                        ${PROJECT_SOURCE_DIR}/pipelines/SingleParticlePipeline.cu)
# cuda_compile_and_embed(embedded_FaradayCage_pipeline
#                        ${PROJECT_SOURCE_DIR}/pipelines/FaradayCagePipeline.cu)
# add_compile_definitions(GPU_SUPPORT)

# add_executable(
#   ViennaPS_GPU ${embedded_SF6O2_pipeline} ${embedded_SingleParticle_pipeline}
#                ${embedded_Fluorocarbon_pipeline} ${PROJECT_SOURCE_DIR}/app/gpuApplication.cpp)
# target_include_directories(ViennaPS_GPU PUBLIC ${OptiX_INCLUDE} ${VIENNAPS_GPU_INCLUDE_DIRS})
# target_link_libraries(ViennaPS_GPU  ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ViennaPS)
# target_compile_definitions(ViennaPS_GPU PUBLIC VIENNACORE_COMPILE_GPU)

# add_dependencies(buildGPUApplication ViennaPS_GPU)

### BENCHMARK ###

# add_executable(ViennaPS_GPU_Benchmark ${embedded_deposition_pipeline}
#                                       ${PROJECT_SOURCE_DIR}/benchmark/GPU_Benchmark.cpp)

# target_include_directories(
#   ViennaPS_GPU_Benchmark PUBLIC ${VIENNAPS_INCLUDE_DIRS} ${OptiX_INCLUDE}
#                                 ${VIENNAPS_GPU_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/app)
# target_link_libraries(ViennaPS_GPU_Benchmark ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ViennaPS)

### EXAMPLES ###

# add_custom_target(buildGPUExamples)

# add_executable(GPU_Trench ${embedded_SingleParticle_pipeline}
#                           ${PROJECT_SOURCE_DIR}/examples/trench.cpp)
# target_include_directories(GPU_Trench PUBLIC ${OptiX_INCLUDE} ${VIENNAPS_GPU_INCLUDE_DIRS})
# target_link_libraries(GPU_Trench ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ViennaPS)
# add_dependencies(buildGPUExamples GPU_Trench)

# add_executable(GPU_Waveguide ${embedded_FaradayCage_pipeline}
#                              ${PROJECT_SOURCE_DIR}/examples/waveguide.cpp)
# target_include_directories(GPU_Waveguide PUBLIC ${OptiX_INCLUDE} ${VIENNAPS_GPU_INCLUDE_DIRS})
# target_link_libraries(GPU_Waveguide ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ViennaPS)
# add_dependencies(buildGPUExamples GPU_Waveguide)

# add_executable(GPU_Hole ${embedded_SF6O2_pipeline}
#                                       ${PROJECT_SOURCE_DIR}/examples/hole.cpp)
# target_include_directories(
#   GPU_Hole PUBLIC ${VIENNAPS_INCLUDE_DIRS} ${OptiX_INCLUDE}
#                                 ${VIENNAPS_GPU_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/app)
# target_link_libraries(GPU_Hole ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${VIENNAPS_LIBRARIES})
# add_dependencies(buildGPUExamples GPU_Hole)

# add_executable(GPU_Custom ${embedded_SingleParticle_pipeline}
#                           ${PROJECT_SOURCE_DIR}/examples/custom.cpp)
# target_include_directories(GPU_Custom PUBLIC ${OptiX_INCLUDE}
#                                              ${VIENNAPS_GPU_INCLUDE_DIRS})
# target_link_libraries(GPU_Custom ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ViennaPS)
# add_dependencies(buildGPUExamples GPU_Custom)

# Now that everything is done, indicate that we have finished configuring at least once.
# We use this variable to set certain defaults only on the first pass, so that we don't
# continually set them over and over again.
set(PASSED_FIRST_CONFIGURE
    ON
    CACHE INTERNAL "Already Configured once?")
