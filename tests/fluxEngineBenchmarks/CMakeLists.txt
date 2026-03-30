project(fluxEngineBenchmarks LANGUAGES CXX)

viennaps_add_executable(plasmaEtchingBenchmark "plasmaEtching.cpp")
viennaps_add_executable(spDepositionBenchmark "spDeposition.cpp")
viennaps_add_executable(spEtchingBenchmark "spEtching.cpp")
viennaps_add_executable(ibeEtchingBenchmark "ibeEtching.cpp")
