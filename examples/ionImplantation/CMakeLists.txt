project(ionImplantation LANGUAGES CXX)

set(VSC_ROOT "")
if(ViennaCS_SOURCE_DIR)
    set(VSC_ROOT "${ViennaCS_SOURCE_DIR}")
elseif(VIENNAPS_VIENNACS_SOURCE)
    set(VSC_ROOT "${VIENNAPS_VIENNACS_SOURCE}")
endif()

# Explicit Pearson IV moments from config
add_executable(${PROJECT_NAME} "ionImplantation.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaPS)
if(VSC_ROOT)
    target_compile_definitions(${PROJECT_NAME} PRIVATE "VSCLIB_DIR=\"${VSC_ROOT}/vsclib\"")
endif()
add_dependencies(ViennaPS_Examples ${PROJECT_NAME})
viennacore_setup_bat(${PROJECT_NAME} ${VIENNAPS_ARTIFACTS_DIRECTORY})

# Table-driven (vsclib database)
add_executable(${PROJECT_NAME}_table "ionImplantation_table.cpp")
target_link_libraries(${PROJECT_NAME}_table PRIVATE ViennaPS)
if(VSC_ROOT)
    target_compile_definitions(${PROJECT_NAME}_table PRIVATE "VSCLIB_DIR=\"${VSC_ROOT}/vsclib\"")
endif()
add_dependencies(ViennaPS_Examples ${PROJECT_NAME}_table)
viennacore_setup_bat(${PROJECT_NAME}_table ${VIENNAPS_ARTIFACTS_DIRECTORY})

# Copy config files into the build directory post-build so they stay synced
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/config.txt"
    "${CMAKE_CURRENT_BINARY_DIR}/config.txt")
add_custom_command(TARGET ${PROJECT_NAME}_table POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/config_table.txt"
    "${CMAKE_CURRENT_BINARY_DIR}/config_table.txt")
