project(voxelCompare LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)
target_compile_definitions(${PROJECT_NAME}
                           PRIVATE VIENNAPS_MECHANISM_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../surfaceChemistry/reactions")

add_dependencies(ViennaPS_Examples ${PROJECT_NAME})

# the same driver in 3D (VC_DIM=3): a trench along y over PMC_YEXT nm, for the
# test whether the 3D cell set reproduces the 2D trench (2026-11-12). Named
# apart from examples/voxelCompare3D, the hole harness of 2026-09.
add_executable(voxelCompareTrench3D "${PROJECT_NAME}.cpp")
target_link_libraries(voxelCompareTrench3D PRIVATE ViennaPS)
target_compile_definitions(voxelCompareTrench3D
                           PRIVATE VIENNAPS_MECHANISM_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../surfaceChemistry/reactions"
                                   VC_DIM=3)
