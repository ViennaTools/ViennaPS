macro(setup_windows_bat TARGET LIB_FOLDER)
  if(WIN32)
    return()
  endif()

  message(STATUS "[ViennaPS] Generating Bat file for ${TARGET}")

  file(
    GENERATE
    OUTPUT "$<TARGET_FILE_DIR:${TARGET}>/${TARGET}.bat"
    CONTENT "set \"PATH=${LIB_FOLDER};%PATH%\"\n${TARGET}.exe %*")
endmacro()
