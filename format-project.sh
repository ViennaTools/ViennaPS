#!/bin/bash

# note: if you aliased clang-format, invoke this script using
# bash -i format-project.sh

find `pwd` -iname "*.hpp" -o -iname "*.cpp" | while read -r i; do clang-format -i "$i"; done
