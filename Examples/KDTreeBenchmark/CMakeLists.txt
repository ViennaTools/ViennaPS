cmake_minimum_required(VERSION 3.4)

project("KDTreeBenchmark")

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${VIENNAPS_LIBRARIES})

add_dependencies(buildExamples ${PROJECT_NAME})
