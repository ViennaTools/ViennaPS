#!/bin/bash

# note: if you aliased clang-format, invoke this script using
# bash -i format-project.sh

find . -type d \( -path ./dependencies -o -path "./build*" \) -prune -false -o -name "*.hpp" -o -name "*.cpp" | while read -r i; do echo "$i"; clang-format -i "$i"; done

# cmake-format can be installed with `pip install --upgrade cmake-format`

if command -v cmake-format &> /dev/null
then
    find . -type d \( -path ./dependencies -o -path "./build*" \) -prune -false -o -name "*.cmake" -o -name CMakeLists.txt -o -name "*.cmake.in"  | while read -r i; do echo "$i"; cmake-format -i "$i"; done
fi
