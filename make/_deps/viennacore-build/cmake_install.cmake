# Install script for directory: /home/filipov/Software/ViennaTools/MaskNoise/ViennaPS/make/_deps/viennacore-src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/lib/llvm-15/bin/llvm-objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/filipov/Software/ViennaTools/MaskNoise/ViennaPS/make/_deps/packageproject-build/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "ViennaCore_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/ViennaCore-1.3.1/ViennaCoreTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/ViennaCore-1.3.1/ViennaCoreTargets.cmake"
         "/home/filipov/Software/ViennaTools/MaskNoise/ViennaPS/make/_deps/viennacore-build/CMakeFiles/Export/57d160db6fe66ecc0928166b7b8d2ea8/ViennaCoreTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/ViennaCore-1.3.1/ViennaCoreTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/ViennaCore-1.3.1/ViennaCoreTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/ViennaCore-1.3.1" TYPE FILE FILES "/home/filipov/Software/ViennaTools/MaskNoise/ViennaPS/make/_deps/viennacore-build/CMakeFiles/Export/57d160db6fe66ecc0928166b7b8d2ea8/ViennaCoreTargets.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "ViennaCore_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/ViennaCore-1.3.1" TYPE FILE FILES
    "/home/filipov/Software/ViennaTools/MaskNoise/ViennaPS/make/_deps/viennacore-build/ViennaCoreConfigVersion.cmake"
    "/home/filipov/Software/ViennaTools/MaskNoise/ViennaPS/make/_deps/viennacore-build/ViennaCoreConfig.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "ViennaCore_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/viennacore-1.3.1" TYPE DIRECTORY FILES "/home/filipov/Software/ViennaTools/MaskNoise/ViennaPS/make/_deps/viennacore-src/include/viennacore/" FILES_MATCHING REGEX "/[^/]*$")
endif()

