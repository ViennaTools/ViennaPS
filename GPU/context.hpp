#pragma once

#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <string>
#include <vector>

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

#include <utLog.hpp>

// global definitions
using NumericType = float;
constexpr int D = 3;

struct pscuContext_t {
  CUmodule getModule(const std::string &moduleName);

  std::vector<std::string> moduleNames;
  std::vector<CUmodule> modules;
};

using pscuContext = pscuContext_t *;

CUmodule pscuContext_t::getModule(const std::string &moduleName) {
  int idx = -1;
  for (int i = 0; i < modules.size(); i++) {
    if (this->moduleNames[i] == moduleName) {
      idx = i;
      break;
    }
  }
  if (idx < 0) {
    std::string modName = moduleName.data();
    utLog::getInstance()
        .addError("Module " + modName + " not in context.")
        .print();
  }

  return modules[idx];
}

void pscuCreateContext(pscuContext &context) {
  // initialize cuda
  cudaFree(0);
  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0) {
    utLog::getInstance().addError("No CUDA capable devices found!").print();
  }

  // init optix
  optixInit();

  context = new pscuContext_t;

  CUmodule normModule;
  CUmodule transModule;
  CUmodule SF6O2ProcessModule;

  CUresult err;

  err = cuInit(0);
  if (err != CUDA_SUCCESS)
    utLog::getInstance().addModuleError("cuInit", err).print();

  std::string normModuleName = "normKernels.ptx";
  err = cuModuleLoad(&normModule, normModuleName.c_str());
  if (err != CUDA_SUCCESS)
    utLog::getInstance().addModuleError(normModuleName, err).print();

  context->moduleNames.push_back(normModuleName);
  context->modules.push_back(normModule);

  std::string transModuleName = "translateKernels.ptx";
  err = cuModuleLoad(&transModule, transModuleName.c_str());
  if (err != CUDA_SUCCESS)
    utLog::getInstance().addModuleError(transModuleName, err).print();

  context->moduleNames.push_back(transModuleName);
  context->modules.push_back(transModule);

  std::string SF6O2ProcessKernelsName = "SF6O2ProcessKernels.ptx";
  err = cuModuleLoad(&SF6O2ProcessModule, SF6O2ProcessKernelsName.c_str());
  if (err != CUDA_SUCCESS)
    utLog::getInstance().addModuleError(SF6O2ProcessKernelsName, err).print();

  context->moduleNames.push_back(SF6O2ProcessKernelsName);
  context->modules.push_back(SF6O2ProcessModule);
}

void pscuReleaseContext(pscuContext context) { delete context; }

void pscuAddModule(const std::string &moduleName, pscuContext context) {
  if (context == nullptr) {
    utLog::getInstance()
        .addError("Context not initialized. Use 'psCreateContext' to "
                  "initialize context.")
        .print();
  }

  CUmodule module;
  CUresult err;

  err = cuModuleLoad(&module, moduleName.c_str());
  if (err != CUDA_SUCCESS) {
    utLog::getInstance().addModuleError(moduleName, err).print();
  }

  context->modules.push_back(module);
  context->moduleNames.push_back(moduleName);
}