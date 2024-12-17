#pragma once

#include <vcLogger.hpp>

#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <string>
#include <vector>

#include <utChecks.hpp>

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

#define STRINGIFY_HELPER(X) #X
#define STRINGIFY(X) STRINGIFY_HELPER(X)

#ifndef VIENNAPS_KERNELS_PATH_DEFINE
#define VIENNAPS_KERNELS_PATH_DEFINE
#endif

#define VIENNAPS_KERNELS_PATH STRINGIFY(VIENNAPS_KERNELS_PATH_DEFINE)

// global definitions
constexpr int DIM = 3;

namespace viennaps {

namespace gpu {

static void contextLogCallback(unsigned int level, const char *tag,
                               const char *message, void *) {
#ifndef NDEBUG
  fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
#endif
}

struct Context_t {
  CUmodule getModule(const std::string &moduleName);

  const std::string modulePath;
  std::vector<std::string> moduleNames;
  std::vector<CUmodule> modules;

  CUcontext cuda;
  cudaDeviceProp deviceProps;
  OptixDeviceContext optix;
  int deviceID;
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

void AddModule(const std::string &moduleName, Context context) {
  if (context == nullptr) {
    viennacore::Logger::getInstance()
        .addError("Context not initialized. Use 'CreateContext' to "
                  "initialize context.")
        .print();
  }

  CUmodule module;
  CUresult err;

  err = cuModuleLoad(&module, (context->modulePath + "/" + moduleName).c_str());
  if (err != CUDA_SUCCESS)
    viennacore::Logger::getInstance().addModuleError(moduleName, err).print();

  context->modules.push_back(module);
  context->moduleNames.push_back(moduleName);
}

void CreateContext(Context &context,
                   std::string modulePath = VIENNAPS_KERNELS_PATH,
                   const int deviceID = 0) {

  // create new context
  context = new Context_t{.modulePath = modulePath, .deviceID = deviceID};

  // initialize CUDA runtime API (cuda## prefix, cuda_runtime_api.h)
  CUDA_CHECK(Free(0));

  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0) {
    viennacore::Logger::getInstance()
        .addError("No CUDA capable devices found!")
        .print();
  }

  cudaSetDevice(deviceID);
  cudaGetDeviceProperties(&context->deviceProps, deviceID);
  viennacore::Logger::getInstance()
      .addDebug("Running on device: " + std::string(context->deviceProps.name))
      .print();

  // initialize CUDA driver API (cu## prefix, cuda.h)
  // we need the CUDA driver API to load kernels from PTX files
  CUresult err;
  err = cuInit(0);
  if (err != CUDA_SUCCESS)
    viennacore::Logger::getInstance().addModuleError("cuInit", err).print();

  err = cuCtxGetCurrent(&context->cuda);
  if (err != CUDA_SUCCESS) {
    viennacore::Logger::getInstance()
        .addError("Error querying current context: error code " +
                  std::to_string(err))
        .print();
  }

  // add default modules
  viennacore::Logger::getInstance()
      .addDebug("PTX kernels path: " + modulePath)
      .print();

  AddModule("normKernels.ptx", context);

  // initialize OptiX context
  OPTIX_CHECK(optixInit());

  optixDeviceContextCreate(context->cuda, 0, &context->optix);
  optixDeviceContextSetLogCallback(context->optix, contextLogCallback, nullptr,
                                   4);
}

void ReleaseContext(Context context) { delete context; }

} // namespace gpu
} // namespace viennaps