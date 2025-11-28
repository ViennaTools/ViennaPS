project(DRAMWiggling LANGUAGES CXX)

viennaps_add_example(${PROJECT_NAME} "${PROJECT_NAME}.cpp")

configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)
# Find all .gds files in the current source directory
file(GLOB GDS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.gds")
# Copy each .gds file to the binary directory
foreach(GDS_FILE ${GDS_FILES})
  get_filename_component(FILENAME ${GDS_FILE} NAME)
  configure_file(${GDS_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} COPYONLY)
endforeach()
