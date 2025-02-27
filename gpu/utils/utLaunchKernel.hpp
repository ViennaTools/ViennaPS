#pragma once

#include <cuda.h>
#include <string>

#include <gpu/vcContext.hpp>

namespace viennaps::gpu {

using namespace viennacore;

// wrapper for launching kernel from a ptx file
class LaunchKernel {
public:
  static void launch(const std::string &moduleName,
                     const std::string &kernelName, void **kernel_args,
                     Context &context, unsigned long sharedMemoryInBytes = 0) {

    CUmodule module = context.getModule(moduleName);
    CUfunction function;

    CUresult err = cuModuleGetFunction(&function, module, kernelName.data());
    if (err != CUDA_SUCCESS)
      viennacore::Logger::getInstance()
          .addFunctionError(std::string(kernelName), err)
          .print();

    err = cuLaunchKernel(function,              // function to call
                         blocks, 1, 1,          /* grid dims */
                         threadsPerBlock, 1, 1, /* block dims */
                         sharedMemoryInBytes * threadsPerBlock, // shared memory
                         0,                                     // stream
                         kernel_args, // kernel parameters
                         nullptr);
    if (err != CUDA_SUCCESS)
      viennacore::Logger::getInstance()
          .addLaunchError(std::string(kernelName), err)
          .print();
  }

  static void launchSingle(const std::string &moduleName,
                           const std::string &kernelName, void **kernel_args,
                           Context &context,
                           unsigned long sharedMemoryInBytes = 0) {

    CUmodule module = context.getModule(moduleName);
    CUfunction function;

    CUresult err = cuModuleGetFunction(&function, module, kernelName.data());
    if (err != CUDA_SUCCESS)
      viennacore::Logger::getInstance()
          .addFunctionError(kernelName, err)
          .print();

    err = cuLaunchKernel(function,            // function to call
                         1, 1, 1,             /* grid dims */
                         1, 1, 1,             /* block dims */
                         sharedMemoryInBytes, // shared memory
                         0,                   // stream
                         kernel_args,         // kernel parameters
                         nullptr);
    if (err != CUDA_SUCCESS)
      viennacore::Logger::getInstance().addLaunchError(kernelName, err).print();
  }

  static constexpr int blocks = 512;
  static constexpr int threadsPerBlock = 512;
};

} // namespace viennaps::gpu
