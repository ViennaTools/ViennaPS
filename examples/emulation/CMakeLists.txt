project(Emulation LANGUAGES CXX)

add_executable(FinFET "FinFET.cpp")
target_link_libraries(FinFET PRIVATE ViennaPS)

viennacore_setup_bat(FinFET ${VIENNAPS_ARTIFACTS_DIRECTORY})
