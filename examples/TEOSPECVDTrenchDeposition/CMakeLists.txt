cmake_minimum_required(VERSION 3.12)
project("TEOS_PE-CVD")

# set default build type
SET(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set (CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH "/home/filipov/Software/ViennaTools/VPS")

find_package(ViennaPS)

# CMake target
add_executable(TEOSPECVD TEOSPECVD.cpp)
target_link_libraries(TEOSPECVD PUBLIC ViennaTools::ViennaPS)
configure_file(TEOSPECVD_config.txt ${CMAKE_CURRENT_BINARY_DIR}/TEOSPECVD_config.txt COPYONLY)
