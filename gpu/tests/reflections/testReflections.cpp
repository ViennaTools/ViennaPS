#include <gpu/vcCudaBuffer.hpp>
#include <rayReflection.hpp>
#include <utLaunchKernel.hpp>

#include <vcRNG.hpp>
#include <vcVectorUtil.hpp>

#include <fstream>
#include <iostream>

#define WRITE_TO_FILE

using namespace viennaps::gpu;

int main() {

  Logger::setLogLevel(LogLevel::DEBUG);

  Context context;
  CreateContext(context);
  const std::string moduleName = "testReflections.ptx";
  AddModule(moduleName, context);

  {
    unsigned numResults = 10000;
    CudaBuffer resultBuffer;
    std::vector<Vec3Df> results(numResults, Vec3Df{0.0f, 0.0f, 0.0f});
    resultBuffer.allocUpload(results);

    auto inDir = Vec3Df{0.0f, 0.0f, -1.0f};
    auto normal = Vec3Df{0.0f, 0.0f, 1.0f};

    CUdeviceptr d_data = resultBuffer.dPointer();
    void *kernel_args[] = {&inDir, &normal, &d_data, &numResults};

    LaunchKernel::launch(moduleName, "test_diffuse", kernel_args, context);

    resultBuffer.download(results.data(), numResults);

#ifdef WRITE_TO_FILE
    std::ofstream file("diffuse_reflection.txt");
    for (auto const &dir : results) {
      file << dir[0] << " " << dir[1] << " " << dir[2] << std::endl;
    }
    file.close();
#endif
  }

  {
    unsigned numResults = 10000;
    CudaBuffer resultBuffer;
    std::vector<Vec3Df> results(numResults, Vec3Df{0.0f, 0.0f, 0.0f});
    resultBuffer.allocUpload(results);

    Vec3Df normal = {0.0, 0.0, 1.0};
    float const minAngle = 85.0 * M_PI / 180.0;
    float incAngle = 45.0 * M_PI / 180.0;
    Vec3Df inDir = {0.0, -std::sin(incAngle), -std::cos(incAngle)};
    float coneAngle = M_PI_2 - std::min(incAngle, minAngle);

    std::cout << "minAngle: " << minAngle << std::endl;
    std::cout << "incAngle: " << incAngle << std::endl;
    std::cout << "incAngle [deg]: " << incAngle * 180.0 / M_PI << std::endl;
    std::cout << "coneAngle: " << coneAngle << std::endl;

    CUdeviceptr d_data = resultBuffer.dPointer();
    void *kernel_args[] = {&inDir, &normal, &coneAngle, &d_data, &numResults};

    LaunchKernel::launch(moduleName, "test_coned_cosine", kernel_args, context);

    resultBuffer.download(results.data(), numResults);

#ifdef WRITE_TO_FILE
    std::ofstream file("coned_specular_reflection.txt");
    for (auto const &dir : results) {
      file << dir[0] << " " << dir[1] << " " << dir[2] << std::endl;
    }
    file.close();
#endif

    {
      std::vector<Vec3Df> directions(numResults);
      viennacore::RNG rngState(21631274);
      // coned specular reflection
      for (int i = 0; i < numResults; ++i) {
        directions[i] = viennaray::ReflectionConedCosine<float, 3>(
            inDir, normal, rngState, coneAngle);
      }

#ifdef WRITE_TO_FILE
      std::ofstream file("coned_specular_reflection_cpu.txt");
      for (auto const &dir : directions) {
        file << dir[0] << " " << dir[1] << " " << dir[2] << std::endl;
      }
      file.close();
#endif
    }
  }
}