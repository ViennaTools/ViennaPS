project(CSVFileProcess LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)

configure_file(rates2D_deposit.csv ${CMAKE_CURRENT_BINARY_DIR}/rates2D_deposit.csv COPYONLY)
configure_file(rates3D_deposit.csv ${CMAKE_CURRENT_BINARY_DIR}/rates3D_deposit.csv COPYONLY)
configure_file(rates2D_etch.csv ${CMAKE_CURRENT_BINARY_DIR}/rates2D_etch.csv COPYONLY)
configure_file(rates3D_etch.csv ${CMAKE_CURRENT_BINARY_DIR}/rates3D_etch.csv COPYONLY)

add_dependencies(ViennaPS_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
