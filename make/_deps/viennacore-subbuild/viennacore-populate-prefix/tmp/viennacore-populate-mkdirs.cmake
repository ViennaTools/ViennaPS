# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/filipov/Software/ViennaTools/MaskNoise/ViennaPS/make/_deps/viennacore-src"
  "/home/filipov/Software/ViennaTools/MaskNoise/ViennaPS/make/_deps/viennacore-build"
  "/home/filipov/Software/ViennaTools/MaskNoise/ViennaPS/make/_deps/viennacore-subbuild/viennacore-populate-prefix"
  "/home/filipov/Software/ViennaTools/MaskNoise/ViennaPS/make/_deps/viennacore-subbuild/viennacore-populate-prefix/tmp"
  "/home/filipov/Software/ViennaTools/MaskNoise/ViennaPS/make/_deps/viennacore-subbuild/viennacore-populate-prefix/src/viennacore-populate-stamp"
  "/home/filipov/Software/ViennaTools/MaskNoise/ViennaPS/make/_deps/viennacore-subbuild/viennacore-populate-prefix/src"
  "/home/filipov/Software/ViennaTools/MaskNoise/ViennaPS/make/_deps/viennacore-subbuild/viennacore-populate-prefix/src/viennacore-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/filipov/Software/ViennaTools/MaskNoise/ViennaPS/make/_deps/viennacore-subbuild/viennacore-populate-prefix/src/viennacore-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/filipov/Software/ViennaTools/MaskNoise/ViennaPS/make/_deps/viennacore-subbuild/viennacore-populate-prefix/src/viennacore-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
