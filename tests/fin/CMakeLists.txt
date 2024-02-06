cmake_minimum_required(VERSION 3.14)

project("fin")

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${VIENNAPS_LIBRARIES})

add_dependencies(buildTests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
set_tests_properties(${PROJECT_NAME} PROPERTIES LABELS "UnitTest")
