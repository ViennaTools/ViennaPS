project(DRAMWiggling LANGUAGES CXX)

if(VIENNAPS_USE_GPU)
  add_gpu_executable(${PROJECT_NAME} target_name "${PROJECT_NAME}.cpp")
else()
  add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
  target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)
endif()

configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)
# Find all .gds files in the current source directory
file(GLOB GDS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.gds")
# Copy each .gds file to the binary directory
foreach(GDS_FILE ${GDS_FILES})
  get_filename_component(FILENAME ${GDS_FILE} NAME)
  configure_file(${GDS_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} COPYONLY)
endforeach()

add_dependencies(ViennaPS_Examples ${PROJECT_NAME})
viennacore_setup_bat(${PROJECT_NAME} ${VIENNAPS_ARTIFACTS_DIRECTORY})
