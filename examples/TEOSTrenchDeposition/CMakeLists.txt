cmake_minimum_required(VERSION 3.4)

project("TEOSTrenchDeposition")

add_executable(singleTEOS singleTEOS.cpp)
target_include_directories(singleTEOS PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(singleTEOS PRIVATE ${VIENNAPS_LIBRARIES})
configure_file(singleTEOS.py ${CMAKE_CURRENT_BINARY_DIR}/singleTEOS.py COPYONLY)
configure_file(singleTEOS_config.txt ${CMAKE_CURRENT_BINARY_DIR}/singleTEOS_config.txt COPYONLY)

add_dependencies(buildExamples singleTEOS)

add_executable(multiTEOS multiTEOS.cpp)
target_include_directories(multiTEOS PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(multiTEOS PRIVATE ${VIENNAPS_LIBRARIES})
configure_file(multiTEOS.py ${CMAKE_CURRENT_BINARY_DIR}/multiTEOS.py COPYONLY)
configure_file(multiTEOS_config.txt ${CMAKE_CURRENT_BINARY_DIR}/multiTEOS_config.txt COPYONLY)

add_dependencies(buildExamples multiTEOS)
