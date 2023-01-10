cmake_minimum_required(VERSION 3.4)

project("InterpolationWorkflow")

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${VIENNAPS_LIBRARIES})

set(CREATE_DATA CreateData)
add_executable(${CREATE_DATA} ${CREATE_DATA}.cpp)
target_include_directories(${CREATE_DATA} PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(${CREATE_DATA} PRIVATE ${VIENNAPS_LIBRARIES})

set(GEOM_RECONSTRUCTION GeometryReconstruction)
add_executable(${GEOM_RECONSTRUCTION} ${GEOM_RECONSTRUCTION}.cpp)
target_include_directories(${GEOM_RECONSTRUCTION} PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(${GEOM_RECONSTRUCTION} PRIVATE ${VIENNAPS_LIBRARIES})

configure_file(data.csv ${CMAKE_CURRENT_BINARY_DIR}/data.csv COPYONLY)


add_dependencies(buildExamples ${PROJECT_NAME} ${CREATE_DATA} ${GEOM_RECONSTRUCTION})
