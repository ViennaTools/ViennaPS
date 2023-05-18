
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was ViennaPSConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)

# ViennaPS requires C++17
set(CMAKE_CXX_STANDARD "17")

# ##################################################################################################
# compiler dependent settings for ViennaPS
# ##################################################################################################
find_dependency(OpenMP)
list(APPEND VIENNAPS_LIBRARIES OpenMP::OpenMP_CXX)

# compiler dependent settings
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # disable-new-dtags sets RPATH which searches for libs recursively, instead of RUNPATH which does
  # not needed for g++ to link correctly
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp /wd\"4267\" /wd\"4244\"")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd\"4267\" /wd\"4244\"")
endif()

set(VIENNAPS_INCLUDE_DIRS "/usr/local/include")

set(ViennaLS_DIR /home/serb/Desktop/ViennaPS/dependencies/Install/viennals_external/lib/cmake/ViennaLS)
set(ViennaRay_DIR /home/serb/Desktop/ViennaPS/dependencies/Install/viennaray_external/lib/cmake/ViennaRay)

find_dependency(ViennaLS PATHS ${ViennaLS_DIR} NO_DEFAULT_PATH)
find_dependency(ViennaRay PATHS ${ViennaRay_DIR} NO_DEFAULT_PATH)

list(APPEND VIENNAPS_INCLUDE_DIRS ${VIENNALS_INCLUDE_DIRS} ${VIENNARAY_INCLUDE_DIRS})
list(APPEND VIENNAPS_LIBRARIES ${VIENNALS_LIBRARIES} ${VIENNARAY_LIBRARIES})

message(STATUS "ViennaLS found at: ${ViennaLS_DIR}")
message(STATUS "ViennaRay found at: ${ViennaRay_DIR}")

if(ON)
  add_compile_definitions(VIENNAPS_VERBOSE)
endif(ON)

check_required_components("ViennaPS")
