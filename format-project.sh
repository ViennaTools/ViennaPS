#!/bin/bash

# note: if you aliased clang-format, invoke this script using
# bash -i format-project.sh

find `pwd` -iname "*.hpp" -o -iname "*.cpp" -not -path "./build*" -not -path "./dependencies/*" | while read -r i; do clang-format -i "$i"; done

# cmake-format can be installed with `pip install --upgrade cmakelang`
if command -v cmake-format &> /dev/null
then
    find `pwd` -iname "CMakeLists.txt" -o -iname "*.cmake.in" -not -path "./build*" -not -path "./dependencies/*"  | while read -r i; do cmake-format --line-width 100 -i "$i"; done
fi
