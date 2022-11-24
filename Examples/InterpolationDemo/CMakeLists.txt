cmake_minimum_required(VERSION 3.4)

project("InterpolationDemo")

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${VIENNAPS_LIBRARIES})

configure_file(griddata.csv ${CMAKE_CURRENT_BINARY_DIR}/griddata.csv COPYONLY)
configure_file(scatterdata.csv ${CMAKE_CURRENT_BINARY_DIR}/scatterdata.csv COPYONLY)

add_dependencies(buildExamples ${PROJECT_NAME})
