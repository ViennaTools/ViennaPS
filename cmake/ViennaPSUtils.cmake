function(viennaps_add_executable target_name source_file)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} PRIVATE ViennaPS)
  if(VIENNAPS_USE_GPU)
    add_dependencies(${target_name} ${VIENNAPS_GPU_DEPENDENCIES})
  endif()
endfunction()

function(viennaps_add_example target_name source_file)
  viennaps_add_executable(${target_name} ${source_file})

  add_dependencies(ViennaPS_Examples ${target_name})
  viennacore_setup_bat(${target_name} ${VIENNAPS_ARTIFACTS_DIRECTORY})
endfunction()
