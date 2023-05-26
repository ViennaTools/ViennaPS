cmake_minimum_required(VERSION 3.4)

project("TEOSTrenchDeposition")

add_executable(SingleTEOS SingleTEOS.cpp)
target_include_directories(SingleTEOS PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(SingleTEOS PRIVATE ${VIENNAPS_LIBRARIES})
configure_file(SingleTEOS_config.txt ${CMAKE_CURRENT_BINARY_DIR}/SingleTEOS_config.txt COPYONLY)

add_dependencies(buildExamples SingleTEOS)

add_executable(MultiTEOS MultiTEOS.cpp)
target_include_directories(MultiTEOS PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(MultiTEOS PRIVATE ${VIENNAPS_LIBRARIES})
configure_file(MultiTEOS_config.txt ${CMAKE_CURRENT_BINARY_DIR}/MultiTEOS_config.txt COPYONLY)

add_dependencies(buildExamples MultiTEOS)
