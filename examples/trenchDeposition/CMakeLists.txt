cmake_minimum_required(VERSION 3.4)

project("trenchDeposition")

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${VIENNAPS_LIBRARIES})
configure_file(trenchDeposition.py ${CMAKE_CURRENT_BINARY_DIR}/trenchDeposition.py COPYONLY)
configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)

add_dependencies(buildExamples ${PROJECT_NAME})
