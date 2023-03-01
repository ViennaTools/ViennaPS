# Install script for directory: /home/serb/Desktop/ViennaPS

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
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
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
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ViennaPS/ViennaPSTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ViennaPS/ViennaPSTargets.cmake"
         "/home/serb/Desktop/ViennaPS/cmake-build-debug/CMakeFiles/Export/eaef28e71c4dc227582c1eb0ea49461c/ViennaPSTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ViennaPS/ViennaPSTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ViennaPS/ViennaPSTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ViennaPS" TYPE FILE FILES "/home/serb/Desktop/ViennaPS/cmake-build-debug/CMakeFiles/Export/eaef28e71c4dc227582c1eb0ea49461c/ViennaPSTargets.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ViennaPS" TYPE FILE FILES
    "/home/serb/Desktop/ViennaPS/cmake-build-debug/ViennaPSConfig.cmake"
    "/home/serb/Desktop/ViennaPS/cmake-build-debug/ViennaPSConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/serb/Desktop/ViennaPS/include/CellSet/csBVH.hpp"
    "/home/serb/Desktop/ViennaPS/include/CellSet/csBoundingVolume.hpp"
    "/home/serb/Desktop/ViennaPS/include/CellSet/csDenseCellSet.hpp"
    "/home/serb/Desktop/ViennaPS/include/CellSet/csTracePath.hpp"
    "/home/serb/Desktop/ViennaPS/include/CellSet/csTracing.hpp"
    "/home/serb/Desktop/ViennaPS/include/CellSet/csTracingKernel.hpp"
    "/home/serb/Desktop/ViennaPS/include/CellSet/csTracingParticle.hpp"
    "/home/serb/Desktop/ViennaPS/include/CellSet/csUtil.hpp"
    "/home/serb/Desktop/ViennaPS/include/Compact/psCSVDataSource.hpp"
    "/home/serb/Desktop/ViennaPS/include/Compact/psCSVReader.hpp"
    "/home/serb/Desktop/ViennaPS/include/Compact/psCSVWriter.hpp"
    "/home/serb/Desktop/ViennaPS/include/Compact/psDataScaler.hpp"
    "/home/serb/Desktop/ViennaPS/include/Compact/psDataSource.hpp"
    "/home/serb/Desktop/ViennaPS/include/Compact/psKDTree.hpp"
    "/home/serb/Desktop/ViennaPS/include/Compact/psNearestNeighborsInterpolation.hpp"
    "/home/serb/Desktop/ViennaPS/include/Compact/psPointLocator.hpp"
    "/home/serb/Desktop/ViennaPS/include/Compact/psQueues.hpp"
    "/home/serb/Desktop/ViennaPS/include/Compact/psRectilinearGridInterpolation.hpp"
    "/home/serb/Desktop/ViennaPS/include/Compact/psValueEstimator.hpp"
    "/home/serb/Desktop/ViennaPS/include/Geometries/psMakeFin.hpp"
    "/home/serb/Desktop/ViennaPS/include/Geometries/psMakeHole.hpp"
    "/home/serb/Desktop/ViennaPS/include/Geometries/psMakePlane.hpp"
    "/home/serb/Desktop/ViennaPS/include/Geometries/psMakeTrench.hpp"
    "/home/serb/Desktop/ViennaPS/include/Models/DirectionalEtching.hpp"
    "/home/serb/Desktop/ViennaPS/include/Models/GeometricDistributionModels.hpp"
    "/home/serb/Desktop/ViennaPS/include/Models/PlasmaDamage.hpp"
    "/home/serb/Desktop/ViennaPS/include/Models/SF6O2Etching.hpp"
    "/home/serb/Desktop/ViennaPS/include/Models/SimpleDeposition.hpp"
    "/home/serb/Desktop/ViennaPS/include/Models/WetEtching.hpp"
    "/home/serb/Desktop/ViennaPS/include/psAdvectionCallback.hpp"
    "/home/serb/Desktop/ViennaPS/include/psDomain.hpp"
    "/home/serb/Desktop/ViennaPS/include/psGDSGeometry.hpp"
    "/home/serb/Desktop/ViennaPS/include/psGDSReader.hpp"
    "/home/serb/Desktop/ViennaPS/include/psGDSUtils.hpp"
    "/home/serb/Desktop/ViennaPS/include/psGeometricModel.hpp"
    "/home/serb/Desktop/ViennaPS/include/psPlanarize.hpp"
    "/home/serb/Desktop/ViennaPS/include/psPointData.hpp"
    "/home/serb/Desktop/ViennaPS/include/psPointValuesToLevelSet.hpp"
    "/home/serb/Desktop/ViennaPS/include/psProcess.hpp"
    "/home/serb/Desktop/ViennaPS/include/psProcessModel.hpp"
    "/home/serb/Desktop/ViennaPS/include/psProcessParams.hpp"
    "/home/serb/Desktop/ViennaPS/include/psSmartPointer.hpp"
    "/home/serb/Desktop/ViennaPS/include/psSurfaceModel.hpp"
    "/home/serb/Desktop/ViennaPS/include/psToDiskMesh.hpp"
    "/home/serb/Desktop/ViennaPS/include/psToSurfaceMesh.hpp"
    "/home/serb/Desktop/ViennaPS/include/psTranslationField.hpp"
    "/home/serb/Desktop/ViennaPS/include/psUtils.hpp"
    "/home/serb/Desktop/ViennaPS/include/psVTKWriter.hpp"
    "/home/serb/Desktop/ViennaPS/include/psVelocityField.hpp"
    "/home/serb/Desktop/ViennaPS/include/psWriteVisualizationMesh.hpp"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/serb/Desktop/ViennaPS/cmake-build-debug/external/upstream/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/serb/Desktop/ViennaPS/cmake-build-debug/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
