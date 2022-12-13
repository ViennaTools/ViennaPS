#pragma once

#include <cuda.h>
#include <string>

#include <context.hpp>
#include <utLog.hpp>

// wrapper for launching kernel from a ptx file
class utLaunchKernel {
public:
  static void launch(const std::string &moduleName,
                     const std::string &kernelName, void **kernel_args,
                     pscuContext_t *context,
                     unsigned long sharedMemoryInBytes = 0) {

    CUmodule module = context->getModule(moduleName);
    CUfunction function;
    CUresult err;

    err = cuModuleGetFunction(&function, module, kernelName.data());
    if (err != CUDA_SUCCESS)
      utLog::getInstance()
          .addFunctionError(std::string(kernelName), err)
          .print();

    err = cuLaunchKernel(function,              // function to call
                         blocks, 1, 1,          /* grid dims */
                         threadsPerBlock, 1, 1, /* block dims */
                         sharedMemoryInBytes * threadsPerBlock, // shared memory
                         0,                                     // stream
                         kernel_args, // kernel parameters
                         NULL);
    if (err != CUDA_SUCCESS)
      utLog::getInstance().addLaunchError(std::string(kernelName), err).print();
  }

  static void launchSingle(const std::string &moduleName,
                           const std::string &kernelName, void **kernel_args,
                           pscuContext_t *context,
                           unsigned long sharedMemoryInBytes = 0) {

    CUmodule module = context->getModule(moduleName);
    CUfunction function;
    CUresult err;

    err = cuModuleGetFunction(&function, module, kernelName.data());
    if (err != CUDA_SUCCESS)
      utLog::getInstance().addFunctionError(kernelName, err).print();

    err = cuLaunchKernel(function,            // function to call
                         1, 1, 1,             /* grid dims */
                         1, 1, 1,             /* block dims */
                         sharedMemoryInBytes, // shared memory
                         0,                   // stream
                         kernel_args,         // kernel parameters
                         NULL);
    if (err != CUDA_SUCCESS)
      utLog::getInstance().addLaunchError(kernelName, err).print();
  }

  static constexpr int blocks = 512;
  static constexpr int threadsPerBlock = 512;
};
