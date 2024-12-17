#pragma once

#include <vcLogger.hpp>

#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <string>
#include <vector>

#include <curtChecks.hpp>

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

// global definitions
constexpr int DIM = 3;

namespace viennaps {

namespace gpu {

struct Context_t {
  CUmodule getModule(const std::string &moduleName);

  const std::string modulePath;
  std::vector<std::string> moduleNames;
  std::vector<CUmodule> modules;
};

using Context = Context_t *;

CUmodule Context_t::getModule(const std::string &moduleName) {
  int idx = -1;
  for (int i = 0; i < modules.size(); i++) {
    if (this->moduleNames[i] == moduleName) {
      idx = i;
      break;
    }
  }
  if (idx < 0) {
    std::string modName = moduleName.data();
    viennacore::Logger::getInstance()
        .addError("Module " + modName + " not in context.")
        .print();
  }

  return modules[idx];
}

void CreateContext(Context &context, std::string modulePath = "lib/ptx/") {
  // initialize cuda
  CUDA_CHECK(Free(0));
  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0) {
    viennacore::Logger::getInstance()
        .addError("No CUDA capable devices found!")
        .print();
  }

  // init optix
  OPTIX_CHECK(optixInit());

  CUmodule normModule;
  // CUmodule transModule;
  // CUmodule SF6O2ProcessModule;
  // CUmodule FluorocarbonProcessModule;

  CUresult err;

  err = cuInit(0);
  if (err != CUDA_SUCCESS)
    viennacore::Logger::getInstance().addModuleError("cuInit", err).print();

  context = new Context_t{.modulePath = modulePath};

  std::string normModuleName = "normKernels";
  err = cuModuleLoad(&normModule, (context->modulePath + "custom_generated_" +
                                   normModuleName + ".cu.ptx")
                                      .c_str());
  if (err != CUDA_SUCCESS)
    viennacore::Logger::getInstance()
        .addModuleError(normModuleName, err)
        .print();

  context->moduleNames.push_back(normModuleName);
  context->modules.push_back(normModule);

  // std::string transModuleName = "translateKernels.ptx";
  // err = cuModuleLoad(&transModule, transModuleName.c_str());
  // if (err != CUDA_SUCCESS)
  //   viennacore::Logger::getInstance().addModuleError(transModuleName,
  //   err).print();

  // context->moduleNames.push_back(transModuleName);
  // context->modules.push_back(transModule);

  // std::string SF6O2ProcessKernelsName = "SF6O2ProcessKernels.ptx";
  // err = cuModuleLoad(&SF6O2ProcessModule, SF6O2ProcessKernelsName.c_str());
  // if (err != CUDA_SUCCESS)
  //   viennacore::Logger::getInstance().addModuleError(SF6O2ProcessKernelsName,
  //   err).print();

  // context->moduleNames.push_back(SF6O2ProcessKernelsName);
  // context->modules.push_back(SF6O2ProcessModule);

  // std::string FluorocarbonProcessKernelsName =
  // "FluorocarbonProcessKernels.ptx"; err =
  // cuModuleLoad(&FluorocarbonProcessModule,
  //                    FluorocarbonProcessKernelsName.c_str());
  // if (err != CUDA_SUCCESS)
  //   viennacore::Logger::getInstance()
  //       .addModuleError(FluorocarbonProcessKernelsName, err)
  //       .print();

  // context->moduleNames.push_back(FluorocarbonProcessKernelsName);
  // context->modules.push_back(FluorocarbonProcessModule);
}

void ReleaseContext(Context context) { delete context; }

void AddModule(const std::string &moduleName, Context context) {
  if (context == nullptr) {
    viennacore::Logger::getInstance()
        .addError("Context not initialized. Use 'psCreateContext' to "
                  "initialize context.")
        .print();
  }

  CUmodule module;
  CUresult err;

  err = cuModuleLoad(&module, moduleName.c_str());
  if (err != CUDA_SUCCESS) {
    viennacore::Logger::getInstance().addModuleError(moduleName, err).print();
  }

  context->modules.push_back(module);
  context->moduleNames.push_back(moduleName);
}

} // namespace gpu
} // namespace viennaps